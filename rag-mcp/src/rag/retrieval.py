import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import ValidationError

from rag.config import Settings
from rag.lexical import search_fts5
from rag.models import Article, ChunkHit, ChunksManifest, Embedder, Reranker
from rag.vector import search_vector

logger = logging.getLogger(__name__)

SearchMethod = Literal['vector', 'bm25', 'hybrid']

_HYBRID_CANDIDATE_POOL = 50


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], rrf_k: int, weights: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for name, ranked_ids in rankings.items():
        weight = 1.0 if weights is None else weights[name]
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    """calc cosine similarity and search over chunk embeddings, optionally fused with BM25 via RRF"""

    @property
    def manifest(self) -> ChunksManifest:
        return self._manifest

    @property
    def categories(self) -> frozenset[str]:
        """Distinct categories present in the loaded chunks for callers that expose them as a filter, ie MCP Server"""
        return frozenset(str(c) for c in self._df['category'].unique())

    def __len__(self) -> int:
        return len(self._df)

    def __init__(
        self,
        df: pd.DataFrame,
        embedder: Embedder,
        manifest: ChunksManifest,
        docs: pd.DataFrame | None = None,
        reranker: Reranker | None = None,
        fts_con: sqlite3.Connection | None = None,
        rrf_k: int = 60,
        fts5_title_weight: float = 10.0,
        fts5_text_weight: float = 1.0,
        rrf_vector_weight: float = 1.0,
        rrf_bm25_weight: float = 1.0,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._docs = docs
        matrix = np.vstack(self._df['embedding'].to_list()).astype(np.float32)  # vert stack matrices
        if not np.isfinite(matrix).all():
            bad_rows = np.where(~np.isfinite(matrix).all(axis=1))[0]
            raise ValueError(f'non-finite embeddings (NaN/inf) at rows : {bad_rows.tolist()}')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # normalize to length 1.0
        if np.any(norms == 0):
            bad_rows = np.where(norms.flatten() == 0)[0]
            raise ValueError(f'zero norm embeddings (divide by zero) at rows : {bad_rows.tolist()}')
        self._matrix = matrix / norms
        self._embedder = embedder
        self._reranker = reranker
        self._manifest = manifest
        self._fts_con = fts_con
        self._rrf_k = rrf_k
        self._fts5_title_weight = fts5_title_weight
        self._fts5_text_weight = fts5_text_weight
        self._rrf_vector_weight = rrf_vector_weight
        self._rrf_bm25_weight = rrf_bm25_weight
        self._chunk_id_to_pos: dict[str, int] = dict(zip(self._df['chunk_id'], range(len(self._df)), strict=True))

    def search(
        self,
        query: str,
        k: int,
        category: str | None = None,
        method: SearchMethod = 'hybrid',
        rerank: bool = False,
        fetch_k: int | None = None,
    ) -> list[ChunkHit]:
        """fetch_k: candidate pool size to fuse/retrieve and rerank before cutting down to k. Only takes effect
        when rerank=True and ignored otherwise since there's nothing to gain from over-fetching without a rerank
        pass to make use of the wider pool. Defaults to k."""
        search_k = max(k, fetch_k) if (rerank and fetch_k is not None) else k

        if method == 'vector':
            ranked = self._search_vector_ranked(query, search_k, category)
        elif self._fts_con is None:
            raise ValueError(f'method={method!r} requires an FTS5 index. None was loaded for this retriever')
        elif method == 'bm25':
            ranked = self._search_bm25_ranked(query, search_k, category)
        else:
            pool = max(search_k, _HYBRID_CANDIDATE_POOL)
            vector_ids = [chunk_id for chunk_id, _ in self._search_vector_ranked(query, pool, category)]
            bm25_ids = [chunk_id for chunk_id, _ in self._search_bm25_ranked(query, pool, category)]
            fused = reciprocal_rank_fusion(
                {'vector': vector_ids, 'bm25': bm25_ids},
                rrf_k=self._rrf_k,
                weights={'vector': self._rrf_vector_weight, 'bm25': self._rrf_bm25_weight},
            )
            ranked = fused[:search_k]

        if rerank:
            if self._reranker is None:
                raise ValueError('rerank=True requires a reranker to be configured on this Retriever')
            ranked = self._rerank(query, ranked)[:k]

        return self._hits_from_ranked(ranked)

    def _search_vector_ranked(self, query: str, k: int, category: str | None) -> list[tuple[str, float]]:
        return search_vector(
            self._matrix,
            self._df['chunk_id'].to_numpy(),
            self._df['category'].to_numpy(),
            self._embedder,
            query,
            k,
            category=category,
        )

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

    def _rerank(self, query: str, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        assert self._reranker is not None  # checked by search() before calling
        ids = [chunk_id for chunk_id, _ in ranked]
        texts = [self._df.iloc[self._chunk_id_to_pos[chunk_id]]['text'] for chunk_id in ids]

        scores = self._reranker.rerank(query, texts)

        return sorted(zip(ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)

    def _hits_from_ranked(self, ranked: list[tuple[str, float]]) -> list[ChunkHit]:
        # mypy/pydantic plugin mistypes to_dict()'s keys as non-str here (but not in unpack in _search_vector);
        # cast is a no op at runtime, keys are str
        hits: list[ChunkHit] = []
        for chunk_id, rank_score in ranked:
            pos = self._chunk_id_to_pos[chunk_id]
            row = cast('dict[str, Any]', self._df.iloc[pos].drop('embedding').to_dict())
            hits.append(ChunkHit(**row, score=rank_score))
        return hits

    def get_article(self, chunk_id: str) -> Article | None:
        """Look up the full source article a chunk was cut from using chunk_id."""
        if self._docs is None:
            raise ValueError('get_article requires a Retriever built with docs (see load_retriever)')
        pos = self._chunk_id_to_pos.get(chunk_id)
        if pos is None:
            return None
        doc_id = self._df.iloc[pos]['doc_id']
        return Article(doc_id=doc_id, **self._docs.loc[doc_id].to_dict())

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """The embedded text of a chunk for locating the chunk inside its source article."""
        pos = self._chunk_id_to_pos.get(chunk_id)
        if pos is None:
            return None
        return cast(str, self._df.iloc[pos]['text'])


class ManifestMismatchError(RuntimeError):
    """Chunks were embedded with a different model or embedding dimension than the current settings."""


class OrphanChunksError(RuntimeError):
    """Chunks reference a doc_id that is missing from the corpus parquet."""


class StaleIndexError(RuntimeError):
    """FTS5 index's chunk_ids doesn't match chunks.parquet's chunk_ids (stale or foreign db)."""


def load_retriever(
    chunks_file: Path, embedder: Embedder, settings: Settings, reranker: Reranker | None = None
) -> Retriever:
    """Load chunks + manifest, merge in document title/url, validate compatibility, return ready Retriever
    Raises:
        FileNotFoundError if the chunks, manifest or corpus file does not exist
        ManifestMismatchError if the manifest is incompatible with the current settings, the corpus on disk
            is not the one the chunks came from (sha256 missmatch), or the manifest predates this build's schema
        OrphanChunksError if a chunk's doc_id has no match in the corpus parquet
        StaleIndexError if the FTS5 index's chunk_ids doesn't match chunks.parquet's chunk_ids
    """

    if not chunks_file.exists():
        raise FileNotFoundError(f'Chunks file not found: {chunks_file}')

    manifest_path = chunks_file.with_suffix('.manifest.json')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest file not found: {manifest_path}')

    try:
        manifest = ChunksManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    except ValidationError as e:
        raise ManifestMismatchError('manifest predates this build — rebuild chunks') from e

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
    fts_con = sqlite3.connect(fts_path, check_same_thread=False)

    docs = pd.read_parquet(
        settings.corpus_path, columns=['doc_id', 'url', 'title', 'category', 'breadcrumb', 'body_md', 'n_chars']
    )
    df = pd.read_parquet(chunks_file).merge(
        docs[['doc_id', 'url', 'title', 'n_chars']].rename(columns={'n_chars': 'full_article_length'}),
        on='doc_id',
        how='left',
        validate='many_to_one',
    )

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

    try:
        fts_ids = {row[0] for row in fts_con.execute('SELECT chunk_id FROM chunks_fts')}
    except sqlite3.Error as e:
        raise StaleIndexError(f'{fts_path} is not a valid FTS5 index ({e}). Rebuild chunks to regenerate it.') from e
    df_ids = set(df['chunk_id'])
    if fts_ids != df_ids:
        missing = sorted(df_ids - fts_ids)[:10]
        extra = sorted(fts_ids - df_ids)[:10]
        raise StaleIndexError(
            f'FTS5 index {fts_path} does not match {chunks_file} ({len(df_ids - fts_ids)} chunk_id(s) missing '
            f'from the index, ex: {missing}; {len(fts_ids - df_ids)} extra in the index, ex: {extra}). '
            'Rebuild chunks to regenerate a matching FTS5 index.'
        )

    return Retriever(
        df,
        embedder,
        manifest,
        docs=docs.set_index('doc_id'),
        reranker=reranker,
        fts_con=fts_con,
        rrf_k=settings.rrf_k,
        fts5_title_weight=settings.fts5_title_weight,
        fts5_text_weight=settings.fts5_text_weight,
        rrf_vector_weight=settings.rrf_vector_weight,
        rrf_bm25_weight=settings.rrf_bm25_weight,
    )
