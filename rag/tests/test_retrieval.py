import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag.config import Settings
from rag.lexical import build_fts5_index
from rag.models import ChunksManifest
from rag.retrieval import (
    ManifestMismatchError,
    OrphanChunksError,
    Retriever,
    StaleIndexError,
    load_retriever,
    reciprocal_rank_fusion,
)


# helpers
class FakeEmbedder:
    """Returns a fixed query vector regardless of input text."""

    def __init__(self, query_vec: list[float]):
        self._vec = np.array(query_vec, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([self._vec], dtype=np.float32)


def _make_manifest(
    model: str = 'Qwen/Qwen3-Embedding-0.6B', dim: int = 2, source_sha256: str = 'abc123'
) -> ChunksManifest:
    return ChunksManifest(
        source_file='fake.parquet',
        source_sha256=source_sha256,
        n_articles=3,
        n_chunks=3,
        min_body_length=100,
        tokenizer_model=model,
        max_tokens=450,
        overlap=50,
        fts5_tokenchar=False,
        parser_version='1',
        embedding_model=model,
        embedding_dim=dim,
        embedding_dtype='float32',
        query_prompt='',
        created_at=datetime.now(UTC),
    )


def _make_chunks_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'chunk_id': ['alpha#000', 'beta#000', 'gamma#000'],
            'doc_id': ['alpha', 'beta', 'gamma'],
            'title': ['Alpha', 'Beta', 'Gamma'],
            'url': [
                'https://example.com/alpha',
                'https://example.com/beta',
                'https://example.com/gamma',
            ],
            'heading_path': [['Alpha'], ['Beta'], ['Gamma']],
            'text': ['Text alpha', 'Text beta', 'Text gamma'],
            'category': ['bestiary', 'bestiary', 'feats'],
            'n_tokens': [10, 9, 10],
            'embedding': [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        }
    )


def _make_settings(model: str = 'Qwen/Qwen3-Embedding-0.6B', dim: int = 2, corpus_path: Path | None = None) -> Settings:
    return Settings(
        embedding_model=model,
        embedding_dim=dim,
        corpus_path=corpus_path or Path('data/corpus.parquet'),
    )


def _write_test_files(tmp_path: Path, model: str, dim: int) -> tuple[Path, Path]:
    docs_df = pd.DataFrame(
        {
            'doc_id': ['alpha', 'beta', 'gamma'],
            'url': [
                'https://example.com/alpha',
                'https://example.com/beta',
                'https://example.com/gamma',
            ],
            'title': ['Alpha', 'Beta', 'Gamma'],
        }
    )
    docs_path = tmp_path / 'corpus.parquet'
    docs_df.to_parquet(docs_path, index=False)

    chunks_df = _make_chunks_df().drop(columns=['title', 'url'])
    chunks_path = tmp_path / 'chunks.parquet'
    chunks_df.to_parquet(chunks_path, index=False)
    manifest = _make_manifest(model=model, dim=dim, source_sha256=hashlib.sha256(docs_path.read_bytes()).hexdigest())
    manifest_path = chunks_path.with_suffix('.manifest.json')
    manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')

    fts_con = sqlite3.connect(chunks_path.with_suffix('.fts5.db'))
    build_fts5_index(chunks_df, fts_con, fts5_tokenchar=False)
    fts_con.close()
    return chunks_path, docs_path


# reciprocal_rank_fusion


def test_single_ranking_preserves_order():
    result = reciprocal_rank_fusion({'vector': ['a', 'b', 'c']}, rrf_k=60)
    assert [item_id for item_id, _ in result] == ['a', 'b', 'c']


def test_single_ranking_scores_match_rrf_formula():
    result = reciprocal_rank_fusion({'vector': ['a', 'b']}, rrf_k=60)
    assert dict(result) == pytest.approx({'a': 1 / 61, 'b': 1 / 62})


def test_item_in_both_rankings_outranks_item_in_one():
    rankings = {'vector': ['a', 'b'], 'bm25': ['a', 'c']}
    result = reciprocal_rank_fusion(rankings, rrf_k=60)
    assert result[0][0] == 'a'
    assert result[0][1] == pytest.approx(2 / 61)


def test_scores_sum_across_rankings():
    rankings = {'vector': ['a', 'b'], 'bm25': ['b', 'a']}
    scores = dict(reciprocal_rank_fusion(rankings, rrf_k=60))
    assert scores['a'] == pytest.approx(1 / 61 + 1 / 62)
    assert scores['b'] == pytest.approx(1 / 62 + 1 / 61)
    assert scores['a'] == pytest.approx(scores['b'])


def test_item_in_only_one_ranking_still_included():
    rankings = {'vector': ['a'], 'bm25': ['b']}
    result = reciprocal_rank_fusion(rankings, rrf_k=60)
    assert {item_id for item_id, _ in result} == {'a', 'b'}


def test_empty_rankings_returns_empty_list():
    assert reciprocal_rank_fusion({}, rrf_k=60) == []


def test_empty_ranked_lists_returns_empty_list():
    assert reciprocal_rank_fusion({'vector': [], 'bm25': []}, rrf_k=60) == []


def test_higher_rrf_k_shrinks_rank_gap_between_top_and_bottom():
    """Larger rrf_k reducess how much rank position matters. The score gap between rank 1 and
    rank 2 should go down as rrf_k goes up."""
    low_k = dict(reciprocal_rank_fusion({'vector': ['a', 'b']}, rrf_k=1))
    high_k = dict(reciprocal_rank_fusion({'vector': ['a', 'b']}, rrf_k=1000))
    assert (low_k['a'] - low_k['b']) > (high_k['a'] - high_k['b'])


# Retriever.search (vector)


def test_ranks_in_expected_order():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=3, method='vector')
    assert [r.doc_id for r in results] == ['alpha', 'beta', 'gamma']


def test_scores_descend():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    results = retriever.search('anything', k=3, method='vector')
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_greater_than_corpus_size_no_crash():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=100, method='vector')
    assert len(results) == 3


def test_non_finite_embedding_raises_at_construction():
    df = _make_chunks_df()
    df.at[1, 'embedding'] = np.array([np.nan, 1.0], dtype=np.float32)  # type: ignore[assignment]  # noqa: PD008
    with pytest.raises(ValueError, match='non-finite'):
        Retriever(df, FakeEmbedder([1.0, 0.0]), _make_manifest())


def test_search_non_finite_query_vector_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([np.nan, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='non finite'):
        retriever.search('anything', k=3, method='vector')


def test_search_zero_norm_query_vector_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([0.0, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='zero-norm'):
        retriever.search('anything', k=3, method='vector')


def test_search_always_returns_k_results():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    assert len(retriever.search('anything', k=2, method='vector')) == 2


def test_hits_carry_doc_id_and_url():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    top = retriever.search('anything', k=1, method='vector')[0]
    assert top.doc_id == 'alpha'
    assert str(top.url) == 'https://example.com/alpha'


# category filter


def test_category_filter_excludes_higher_scoring_other_category():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=3, category='feats', method='vector')
    assert [r.doc_id for r in results] == ['gamma']


def test_category_filter_k_larger_than_category_size_no_inf_filler():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=10, category='feats', method='vector')
    assert len(results) == 1
    assert results[0].doc_id == 'gamma'


# Retriever.search (bm25 / hybrid)


def _make_retriever_with_fts(query_vec: list[float], rrf_k: int = 60) -> Retriever:
    chunks_df = _make_chunks_df()
    fts_con = sqlite3.connect(':memory:')
    build_fts5_index(chunks_df, fts_con, fts5_tokenchar=False)
    return Retriever(chunks_df, FakeEmbedder(query_vec), _make_manifest(), fts_con=fts_con, rrf_k=rrf_k)


def test_bm25_method_without_fts_index_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='FTS5 index'):
        retriever.search('anything', k=3, method='bm25')


def test_hybrid_method_without_fts_index_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='FTS5 index'):
        retriever.search('anything', k=3, method='hybrid')


def test_bm25_search_returns_matching_chunk():
    # chunk text is 'Text alpha', 'Text beta' and 'Text gamma'. 'gamma' matches only the gamma chunk
    retriever = _make_retriever_with_fts([1.0, 0.0])
    results = retriever.search('gamma', k=3, method='bm25')
    assert [r.doc_id for r in results] == ['gamma']


def test_bm25_search_ignores_query_embedding():
    """method='bm25' must rank with the FTS5 match not the (fake) query embedding result which is set up to favor alpha
    from cosine similarity."""
    retriever = _make_retriever_with_fts([1.0, 0.0])
    results = retriever.search('gamma', k=3, method='bm25')
    assert results[0].doc_id == 'gamma'


def test_hybrid_is_default_method():
    retriever = _make_retriever_with_fts([1.0, 0.0])
    default_call = retriever.search('gamma', k=3)
    explicit_hybrid = retriever.search('gamma', k=3, method='hybrid')
    assert [r.doc_id for r in default_call] == [r.doc_id for r in explicit_hybrid]


def test_hybrid_search_promotes_bm25_match_that_vector_ranked_last():
    """Gamma ranks last by  cosine similarity but is the only bm25 match for 'gamma'.
    Hybrid should pull it back up to first result."""
    retriever = _make_retriever_with_fts([1.0, 0.0])
    vector_only = retriever.search('gamma', k=3, method='vector')
    assert vector_only[0].doc_id == 'alpha'

    hybrid = retriever.search('gamma', k=3, method='hybrid')
    assert hybrid[0].doc_id == 'gamma'


def test_hybrid_search_category_filter_excludes_bm25_only_match():
    # gamma is in category 'feats'. Filtering to 'bestiary' must exclude gamma even though bm25 returns it as top result
    retriever = _make_retriever_with_fts([1.0, 0.0])
    results = retriever.search('gamma', k=3, method='hybrid', category='bestiary')
    assert 'gamma' not in [r.doc_id for r in results]


# load_retriever


def test_load_retriever_merges_title_and_url_from_documents(tmp_path):
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    retriever = load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)
    top = retriever.search('anything', k=1, method='vector')[0]
    assert top.doc_id == 'alpha'
    assert top.title == 'Alpha'
    assert str(top.url) == 'https://example.com/alpha'


def test_manifest_model_mismatch(tmp_path):
    """raise ManifestMismatchError on setting mismatch with manifest (model)"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='wrong-model', dim=2)
    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(ManifestMismatchError, match='embedding model'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)


def test_manifest_dim_mismatch(tmp_path):
    """raise ManifestMismatchError on setting mismatch with manifest (dim)"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=999)
    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(ManifestMismatchError, match='embedding dim'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)


def test_corpus_sha256_drift(tmp_path):
    """raise ManifestMismatchError when corpus.parquet was rewritten after the chunks were created from it"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    docs_df = pd.read_parquet(docs_path)
    docs_df.loc[0, 'title'] = 'Alpha (edited)'
    docs_df.to_parquet(docs_path, index=False)

    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(ManifestMismatchError, match='sha256'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)


def test_manifest_predates_schema_raises_manifest_mismatch(tmp_path):
    """A manifest written before fts5_tokenchar was added should raise ManifestMismatchError, not ValidationError"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    manifest_path = chunks_path.with_suffix('.manifest.json')
    stale_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    del stale_manifest['fts5_tokenchar']
    manifest_path.write_text(json.dumps(stale_manifest), encoding='utf-8')

    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(ManifestMismatchError, match='predates this build'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)


def test_orphan_chunk_fails_at_load(tmp_path):
    """A chunk with it's doc_id missing from corpus.parquet should fail to load vs causing issues in later search"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    chunks_df = pd.read_parquet(chunks_path)
    orphan_row = chunks_df.iloc[[0]].copy()
    orphan_row['chunk_id'] = 'orphan#000'
    orphan_row['doc_id'] = 'orphan'
    pd.concat([chunks_df, orphan_row], ignore_index=True).to_parquet(chunks_path, index=False)

    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(OrphanChunksError, match='orphan'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)


def test_fts5_index_different_ids_fails_at_load(tmp_path):
    """Chunks.parquet that has drifted from its chunks.fts5.db should fail to load vs causing KeyError at query time"""
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    fts_con = sqlite3.connect(chunks_path.with_suffix('.fts5.db'))
    fts_con.execute("UPDATE chunks_fts SET chunk_id = 'swapped#000' WHERE chunk_id = 'alpha#000'")
    fts_con.commit()
    fts_con.close()

    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    with pytest.raises(StaleIndexError, match='FTS5'):
        load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)
