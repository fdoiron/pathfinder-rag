import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rag.config import Settings
from rag.models import Article, CorpusManifest, Embedder, SearchResult

logger = logging.getLogger(__name__)

# TODO: filtering by type of articles type


class Retriever:
    """calc cosine similarity and search over corpus embeddings"""

    @property
    def manifest(self) -> CorpusManifest:
        return self._manifest

    def __init__(self, df: pd.DataFrame, embedder: Embedder, manifest: CorpusManifest) -> None:
        self._df = df.reset_index(drop=True)
        matrix = np.vstack(df['embedding'].to_list()).astype(np.float32)  # vert stack matrices
        if not np.isfinite(matrix).all():
            bad_rows = np.where(~np.isfinite(matrix).all(axis=1))[0]
            raise ValueError(f'non-finite embeddings (NaN/inf) at rows : {bad_rows.tolist()}')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # normalize to length 1.0
        if np.any(norms == 0):
            bad_rows = np.where(norms.flatten() == 0)[0]
            raise ValueError(f'zero norm embeddings (divide by zero) at rows : {bad_rows.tolist()}')
        self._matrix = matrix / norms
        self._embedder = embedder
        self._manifest = manifest

    def search(self, query: str, k: int) -> list[SearchResult]:
        q = self._embedder.embed(
            [query],
            task_type='RETRIEVAL_QUERY',
        )[0]  # embed search query
        q = q / np.linalg.norm(q)  # normalize to length 1.0
        scores = self._matrix @ q  # dot product normalize = cosine similarity
        top_results = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(article=Article(**self._df.iloc[res].drop('embedding').to_dict()), score=float(scores[res]))
            for res in top_results
        ]


class ManifestMismatchError(RuntimeError):
    """Corpus was embedded with a different model or embedding dimension than the current settings."""


def load_retriever(embedding_file: Path, embedder: Embedder, settings: Settings) -> Retriever:
    """Load corpus + manifest, validate compatibility, return ready Retriever
    Raises:
        FileNotFoundError if either embedding or manifest file does not exist
        ManifestMismatchError if the manifest is incompatible with the current settings
    """

    if not embedding_file.exists():
        raise FileNotFoundError(f'Embedding file not found: {embedding_file}')

    manifest_path = embedding_file.with_suffix('.manifest.json')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest file not found: {manifest_path}')

    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))

    if manifest.embedding_model != settings.embedding_model:
        raise ManifestMismatchError(
            f'Manifest embedding model "{manifest.embedding_model}" does not match '
            f'configured model "{settings.embedding_model}"'
        )
    if manifest.embedding_dim != settings.embedding_dim:
        raise ManifestMismatchError(
            f'Manifest embedding dim {manifest.embedding_dim} does not match configured dim {settings.embedding_dim}'
        )

    df = pd.read_parquet(embedding_file)
    # bugfix: save parquet and load turns None into NaN -> breaks pydantic model. rebuild articles excluding embeddings

    metadata_cols = [c for c in df.columns if c != 'embedding']
    meta = df[metadata_cols].astype(object)
    df[metadata_cols] = meta.where(meta.notna(), None)  # replace NaN with None for pydantic model
    logger.info(f'Loaded {len(df)} articles from {embedding_file}')

    return Retriever(df, embedder, manifest)
