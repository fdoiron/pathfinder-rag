from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag.config import Settings
from rag.models import CorpusManifest
from rag.retrieval import ManifestMismatchError, Retriever, load_retriever


# helpers
class FakeEmbedder:
    """Returns a fixed query vector regardless of input text."""

    def __init__(self, query_vec: list[float]):
        self._vec = np.array(query_vec, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:
        return np.array([self._vec], dtype=np.float32)


def _make_manifest(model: str = 'gemini-embedding-001', dim: int = 2) -> CorpusManifest:
    return CorpusManifest(
        source_file='fake.parquet',
        source_sha256='abc123',
        n_articles=3,
        text_columns=['title', 'body'],
        embedding_model=model,
        embedding_dim=dim,
        task_type='RETRIEVAL_DOCUMENT',
        created_at=datetime.now(UTC),
    )


def _make_corpus_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'title': ['Alpha', 'Beta', 'Gamma'],
            'url': [
                'https://example.com/alpha',
                'https://example.com/beta',
                'https://example.com/gamma',
            ],
            'body': ['Body alpha', 'Body beta', 'Body gamma'],
            'embedding': [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([-1.0, 0.0], dtype=np.float32),
            ],
        }
    )


def _make_settings(model: str = 'gemini-embedding-001', dim: int = 2) -> Settings:
    return Settings(
        gcp_service_account_file=None,
        gcp_project='test-proj',
        embedding_model=model,
        embedding_dim=dim,
    )


def _write_test_files(tmp_path: Path, model: str, dim: int) -> tuple[Path, Path]:
    parquet_path = tmp_path / 'corpus.parquet'
    _make_corpus_df().to_parquet(parquet_path, index=False)
    manifest_path = parquet_path.with_suffix('.manifest.json')
    manifest_path.write_text(_make_manifest(model=model, dim=dim).model_dump_json(), encoding='utf-8')
    return parquet_path, manifest_path


# tests
def test_top1_known_nearest_vector():
    """must rank Alpha first"""
    retriever = Retriever(_make_corpus_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    results = retriever.search('anything', k=3)
    assert results[0].article.title == 'Alpha'


def test_scores_descend():
    """results return by descending score"""
    retriever = Retriever(_make_corpus_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    results = retriever.search('anything', k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_greater_than_corpus_size_no_crash():
    """k larger than corpus yields all rows"""
    retriever = Retriever(_make_corpus_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())
    results = retriever.search('anything', k=100)
    assert len(results) == 3


def test_non_finite_embedding_raises_at_construction():
    """corrupt corpus (NaN embedding) fails at load"""
    df = _make_corpus_df()
    df.at[1, 'embedding'] = np.array([np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match='non-finite'):
        Retriever(df, FakeEmbedder([1.0, 0.0]), _make_manifest())


def test_search_always_returns_k_results():
    """valid corpus returns exactly k results never silently fewer"""
    retriever = Retriever(_make_corpus_df(), FakeEmbedder([1.0, 0.01]), _make_manifest())
    assert len(retriever.search('anything', k=2)) == 2


def test_manifest_model_mismatch(tmp_path):
    """raise ManifestMismatchError on setting mismatch with manifest (model)"""
    parquet_path, _ = _write_test_files(tmp_path, model='wrong-model', dim=2)
    settings = _make_settings(model='gemini-embedding-001', dim=2)
    with pytest.raises(ManifestMismatchError, match='embedding model'):
        load_retriever(parquet_path, FakeEmbedder([1.0, 0.0]), settings)


def test_manifest_dim_mismatch(tmp_path):
    """raise ManifestMismatchError on setting mismatch with manifest (dim)"""
    parquet_path, _ = _write_test_files(tmp_path, model='gemini-embedding-001', dim=999)
    settings = _make_settings(model='gemini-embedding-001', dim=2)
    with pytest.raises(ManifestMismatchError, match='embedding dim'):
        load_retriever(parquet_path, FakeEmbedder([1.0, 0.0]), settings)
