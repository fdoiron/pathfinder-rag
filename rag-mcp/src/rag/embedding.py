from typing import cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag.config import Settings
from rag.models import TaskType


class EmbedderUnavailableError(RuntimeError):
    """The embedding model could not be loaded."""


class LocalEmbedder:
    """Embeds via local model in-process"""

    def __init__(self, settings: Settings):
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self._dtype = str(dtype)
        self._model = SentenceTransformer(settings.embedding_model, model_kwargs={'torch_dtype': dtype})
        if 'query' not in self._model.prompts:
            raise ValueError(
                f"model {settings.embedding_model!r} has no 'query' prompt defined. "
                'Query embedding will fail at call time'
            )
        self._batch_size = settings.embedding_batch_size
        self._dim = settings.embedding_dim

    @property
    def query_prompt(self) -> str:
        """Instruction text prepended to queries for manifest"""
        return cast(str, self._model.prompts.get('query', ''))

    @property
    def torch_dtype(self) -> str:
        return self._dtype

    def embed(self, texts: list[str], task_type: TaskType = 'RETRIEVAL_DOCUMENT') -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
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


def load_embedder(settings: Settings) -> LocalEmbedder:
    try:
        return LocalEmbedder(settings)
    except ValueError as e:  # unusable model. Malformed repo id (HFValidationError), or no query prompt
        raise EmbedderUnavailableError(f'Cannot use embedding model {settings.embedding_model!r}: {e}') from e
    except OSError as e:  # hub and cache failures are OSErrors
        raise EmbedderUnavailableError(
            f'Cannot load embedding model {settings.embedding_model!r}.Cause: {type(e).__name__}: {e}'
        ) from e
