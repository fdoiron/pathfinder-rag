import logging
import time
from typing import cast

import httpx
import numpy as np
import torch
from google import genai
from google.genai import errors, types
from sentence_transformers import SentenceTransformer

from rag.config import Settings
from rag.models import TaskType

logger = logging.getLogger(__name__)


RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _is_retryable(e: Exception) -> bool:
    """rate limits server issues retry, auth/bad request fail immediately"""
    if isinstance(e, errors.APIError):
        return e.code in RETRYABLE_CODES
    # network level issues
    return isinstance(e, (httpx.TimeoutException, httpx.TransportError, ConnectionError))


class VertexEmbedder:
    """embeds via VertexAI API"""

    def __init__(self, client: genai.Client, settings: Settings):
        self._client = client
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size

    def embed(self, texts: list[str], task_type: TaskType = 'RETRIEVAL_DOCUMENT') -> np.ndarray:
        """embeds text, returns float32 array (N, dim) in input order"""

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch, task_type=task_type))
            logger.info(f'embedded {min(start + self._batch_size, len(texts))}/{len(texts)}')
        return np.array(vectors, dtype=np.float32)

    def _call_api(self, batch: list[str], task_type: TaskType) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim),
        )
        if response.embeddings is None:
            raise RuntimeError(f'embedding failed, no embeddings returned: {response}')
        vectors = [list(vals) for emb in response.embeddings if (vals := emb.values) is not None]
        if len(vectors) != len(batch):
            raise RuntimeError(f'embedding returned {len(vectors)} vectors for {len(batch)} inputs')
        return vectors

    def _embed_batch(self, batch: list[str], task_type: TaskType, retries: int = 3) -> list[list[float]]:
        for attempt in range(retries):
            try:
                return self._call_api(batch, task_type)
            except Exception as e:
                if not _is_retryable(e) or attempt == retries - 1:
                    raise
                wait = 2**attempt
                logger.warning(f'embedding failed, retry in {wait} seconds: {e}')
                time.sleep(wait)
        raise RuntimeError(f'embedding failed after {retries} retries')


class LocalEmbedder:
    """Embeds via local model in-process"""

    def __init__(self, settings: Settings):
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self._model = SentenceTransformer(settings.embedding_model, model_kwargs={'torch_dtype': dtype})
        if 'query' not in self._model.prompts:
            raise ValueError(
                f"model {settings.embedding_model!r} has no 'query' prompt defined — "
                'query embedding will fail at call time'
            )
        self._batch_size = settings.embedding_batch_size
        self._dim = settings.embedding_dim

    @property
    def query_prompt(self) -> str:
        """Instruction text prepended to queries for manifest"""
        return cast(str, self._model.prompts.get('query', ''))

    def embed(self, texts: list[str], task_type: TaskType = 'RETRIEVAL_DOCUMENT') -> np.ndarray:
        vectors = self._model.encode(
            texts,
            prompt_name='query' if task_type == 'RETRIEVAL_QUERY' else None,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # required: retriever uses dot-product as cosine sim.
            show_progress_bar=len(texts) > self._batch_size,
        )
        if vectors.shape[1] != self._dim:
            raise ValueError(f'model produced {vectors.shape[1]}-dim vectors but settings.embedding_dim is {self._dim}')
        return cast(np.ndarray, vectors.astype(np.float32))
