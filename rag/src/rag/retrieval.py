import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from rag.config import Settings
from rag.lexical import search_fts5
from rag.models import ChunkHit, ChunksManifest, Embedder

logger = logging.getLogger(__name__)

SearchMethod = Literal['vector', 'bm25', 'hybrid']

_HYBRID_CANDIDATE_POOL = 50


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_ids in rankings.values():
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    """calc cosine similarity and search over chunk embeddings, optionally fused with BM25 via RRF"""

    @property
    def manifest(self) -> ChunksManifest:
        return self._manifest

    def __len__(self) -> int:
        return len(self._df)

    def __init__(
        self,
        df: pd.DataFrame,
        embedder: Embedder,
        manifest: ChunksManifest,
        fts_con: sqlite3.Connection | None = None,
        rrf_k: int = 60,
        fts5_title_weight: float = 10.0,
        fts5_text_weight: float = 1.0,
    ) -> None:
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
        self._fts_con = fts_con
        self._rrf_k = rrf_k
        self._fts5_title_weight = fts5_title_weight
        self._fts5_text_weight = fts5_text_weight
        self._chunk_id_to_pos: dict[str, int] = dict(zip(self._df['chunk_id'], range(len(self._df)), strict=True))

    def search(
        self, query: str, k: int, category: str | None = None, method: SearchMethod = 'hybrid'
    ) -> list[ChunkHit]:
        if method == 'vector':
            return self._search_vector(query, k, category)

        if self._fts_con is None:
            raise ValueError(f'method={method!r} requires an FTS5 index. None was loaded for this retriever')

        if method == 'bm25':
            ranked = self._search_bm25_ranked(query, k, category)
            return self._hits_from_ranked(ranked)

        pool = max(k, _HYBRID_CANDIDATE_POOL)
        vector_ids = [hit.chunk_id for hit in self._search_vector(query, pool, category)]
        bm25_ids = [chunk_id for chunk_id, _ in self._search_bm25_ranked(query, pool, category)]
        fused = reciprocal_rank_fusion({'vector': vector_ids, 'bm25': bm25_ids}, k=self._rrf_k)
        return self._hits_from_ranked(fused[:k])

    def _search_vector(self, query: str, k: int, category: str | None) -> list[ChunkHit]:
        q = self._embedder.embed(
            [query],
            task_type='RETRIEVAL_QUERY',
        )[0]  # embed search query
        if not np.isfinite(q).all():
            raise ValueError(f'embedder returned a non finite (NaN/inf) vector for query {query!r}')
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError(f'embedder returned a zero-norm vector for query {query!r}')
        q = q / q_norm  # normalize to length 1.0
        scores = self._matrix @ q  # dot product normalize = cosine similarity
        if category is not None:
            scores = np.where(self._df['category'].to_numpy() == category, scores, -np.inf)
        top_results = [i for i in np.argsort(scores)[::-1][:k] if scores[i] > -np.inf]
        return [
            ChunkHit(**self._df.iloc[res].drop('embedding').to_dict(), score=float(scores[res])) for res in top_results
        ]

    def _search_bm25_ranked(self, query: str, k: int, category: str | None) -> list[tuple[str, float]]:
        assert self._fts_con is not None  # checked by search() before calling
        return search_fts5(
            self._fts_con,
            query,
            k=k,
            title_weight=self._fts5_title_weight,
            text_weight=self._fts5_text_weight,
            fts5_tokenchar=self._manifest.fts5_tokenchar,
            category=category,
        )

    def _hits_from_ranked(self, ranked: list[tuple[str, float]]) -> list[ChunkHit]:
        # bug fix: mypy/pydantic plugin mistypes to_dict()'s keys as non-str here (but not in unpack in _search_vector)
        # cast is a no op at runtime, keys are str
        hits: list[ChunkHit] = []
        for chunk_id, rank_score in ranked:
            pos = self._chunk_id_to_pos[chunk_id]
            row = cast('dict[str, Any]', self._df.iloc[pos].drop('embedding').to_dict())
            hits.append(ChunkHit(**row, score=rank_score))
        return hits


class ManifestMismatchError(RuntimeError):
    """Chunks were embedded with a different model or embedding dimension than the current settings."""


class OrphanChunksError(RuntimeError):
    """Chunks reference a doc_id that is missing from the corpus parquet."""


def load_retriever(chunks_file: Path, embedder: Embedder, settings: Settings) -> Retriever:
    """Load chunks + manifest, merge in document title/url, validate compatibility, return ready Retriever
    Raises:
        FileNotFoundError if the chunks, manifest or corpus file does not exist
        ManifestMismatchError if the manifest is incompatible with the current settings, or the corpus on disk
            is not the one the chunks came from (sha256 missmatch)
        OrphanChunksError if a chunk's doc_id has no match in the corpus parquet
    """

    if not chunks_file.exists():
        raise FileNotFoundError(f'Chunks file not found: {chunks_file}')

    manifest_path = chunks_file.with_suffix('.manifest.json')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest file not found: {manifest_path}')

    manifest = ChunksManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))

    if manifest.embedding_model != settings.embedding_model:
        raise ManifestMismatchError(
            f'Manifest embedding model "{manifest.embedding_model}" does not match '
            f'configured model "{settings.embedding_model}"'
        )
    if manifest.embedding_dim != settings.embedding_dim:
        raise ManifestMismatchError(
            f'Manifest embedding dim {manifest.embedding_dim} does not match configured dim {settings.embedding_dim}'
        )

    if not settings.corpus_path.exists():
        raise FileNotFoundError(f'Corpus file not found: {settings.corpus_path}')
    corpus_sha256 = hashlib.sha256(settings.corpus_path.read_bytes()).hexdigest()
    if corpus_sha256 != manifest.source_sha256:
        raise ManifestMismatchError(
            f'Corpus {settings.corpus_path} (sha256 {corpus_sha256[:12]}) is not the file these chunks came from '
            f'(manifest source_file "{manifest.source_file}", sha256 {manifest.source_sha256[:12]}). '
            'Rebuild chunks against the current corpus.'
        )

    fts_path = chunks_file.with_suffix('.fts5.db')
    if not fts_path.exists():
        raise FileNotFoundError(f'FTS5 index not found: {fts_path}. Rebuild chunks to generate it.')
    # TODO: the fts5 db is only checked for existence, not staleness.  Add a check once
    # df is loaded below: SELECT count(*) FROM chunks_fts vs len(df), raise ManifestMismatchError.
    fts_con = sqlite3.connect(fts_path)

    docs = pd.read_parquet(settings.corpus_path, columns=['doc_id', 'url', 'title'])
    df = pd.read_parquet(chunks_file).merge(docs, on='doc_id', how='left', validate='many_to_one')

    orphan_doc_ids = sorted(df.loc[df['title'].isna(), 'doc_id'].unique())
    if orphan_doc_ids:
        sample = orphan_doc_ids[:10]
        more = ' ...' if len(orphan_doc_ids) > 10 else ''
        raise OrphanChunksError(
            f'{len(orphan_doc_ids)} doc_id(s) in {chunks_file} have no match in '
            f'{settings.corpus_path}: {sample}{more}. Rebuild chunks against the current corpus.'
        )

    # bugfix: save parquet and load turns None into NaN -> breaks pydantic model. rebuild chunks excluding embeddings

    metadata_cols = [c for c in df.columns if c != 'embedding']
    meta = df[metadata_cols].astype(object)
    df[metadata_cols] = meta.where(meta.notna(), None)  # replace NaN with None for pydantic model
    logger.info(f'Loaded {len(df)} chunks from {chunks_file}')

    return Retriever(
        df,
        embedder,
        manifest,
        fts_con=fts_con,
        rrf_k=settings.rrf_k,
        fts5_title_weight=settings.fts5_title_weight,
        fts5_text_weight=settings.fts5_text_weight,
    )
