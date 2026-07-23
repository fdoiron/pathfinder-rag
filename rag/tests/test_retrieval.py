from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag.config import Settings
from rag.models import ChunksManifest
from rag.retrieval import ManifestMismatchError, Retriever, load_retriever


# helpers
class FakeEmbedder:
    """Returns a fixed query vector regardless of input text."""

    def __init__(self, query_vec: list[float]):
        self._vec = np.array(query_vec, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([self._vec], dtype=np.float32)


def _make_manifest(model: str = 'Qwen/Qwen3-Embedding-0.6B', dim: int = 2) -> ChunksManifest:
    return ChunksManifest(
        source_file='fake.parquet',
        source_sha256='abc123',
        n_articles=3,
        n_chunks=3,
        min_body_length=100,
        tokenizer_model=model,
        max_tokens=450,
        overlap=50,
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
        gcp_service_account_file=None,
        gcp_project='test-proj',
        embedding_model=model,
        embedding_dim=dim,
        corpus_path=corpus_path or Path('data/corpus.parquet'),
    )


def _write_test_files(tmp_path: Path, model: str, dim: int) -> tuple[Path, Path]:
    chunks_df = _make_chunks_df().drop(columns=['title', 'url'])
    chunks_path = tmp_path / 'chunks.parquet'
    chunks_df.to_parquet(chunks_path, index=False)
    manifest_path = chunks_path.with_suffix('.manifest.json')
    manifest_path.write_text(_make_manifest(model=model, dim=dim).model_dump_json(), encoding='utf-8')

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
    return chunks_path, docs_path


def test_ranks_in_expected_order():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=3)
    assert [r.doc_id for r in results] == ['alpha', 'beta', 'gamma']


def test_scores_descend():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    results = retriever.search('anything', k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_greater_than_corpus_size_no_crash():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=100)
    assert len(results) == 3


def test_non_finite_embedding_raises_at_construction():
    df = _make_chunks_df()
    df.at[1, 'embedding'] = np.array([np.nan, 1.0], dtype=np.float32)  # noqa: PD008
    with pytest.raises(ValueError, match='non-finite'):
        Retriever(df, FakeEmbedder([1.0, 0.0]), _make_manifest())


def test_search_non_finite_query_vector_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([np.nan, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='non finite'):
        retriever.search('anything', k=3)


def test_search_zero_norm_query_vector_raises():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([0.0, 0.0]), _make_manifest())
    with pytest.raises(ValueError, match='zero-norm'):
        retriever.search('anything', k=3)


def test_search_always_returns_k_results():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    assert len(retriever.search('anything', k=2)) == 2


def test_hits_carry_doc_id_and_url():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    top = retriever.search('anything', k=1)[0]
    assert top.doc_id == 'alpha'
    assert str(top.url) == 'https://example.com/alpha'


# category filter


def test_category_filter_excludes_higher_scoring_other_category():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=3, category='feats')
    assert [r.doc_id for r in results] == ['gamma']


def test_category_filter_k_larger_than_category_size_no_inf_filler():
    retriever = Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=10, category='feats')
    assert len(results) == 1
    assert results[0].doc_id == 'gamma'


# load_retriever


def test_load_retriever_merges_title_and_url_from_documents(tmp_path):
    chunks_path, docs_path = _write_test_files(tmp_path, model='Qwen/Qwen3-Embedding-0.6B', dim=2)
    settings = _make_settings(model='Qwen/Qwen3-Embedding-0.6B', dim=2, corpus_path=docs_path)
    retriever = load_retriever(chunks_path, FakeEmbedder([1.0, 0.0]), settings)
    top = retriever.search('anything', k=1)[0]
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
