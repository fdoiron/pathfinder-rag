import numpy as np
import pandas as pd
import pytest
from google.genai import errors

from rag.corpus import embed_corpus
from rag.embedding import VertexEmbedder


class FakeEmbedder:
    """vector[0] -> input position, verify order was preserved after batching"""

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([[float(i), float(len(t))] for i, t in enumerate(texts)], dtype=np.float32)


def test_embed_corpus_concats_and_preserve_rows():
    df = pd.DataFrame(
        {
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'body': ['Body 1', 'Body 2', 'Body 3'],
            'url': ['URL 1', 'URL 2', 'URL 3'],  # not required to embed
        }
    )
    out = embed_corpus(df, FakeEmbedder(), text_columns=['title', 'body'])

    assert len(out) == 3
    assert list(out['url']) == ['URL 1', 'URL 2', 'URL 3']

    for i, vector in enumerate(out['embedding']):
        assert vector[0] == i  # first element is the input position
        assert vector[1] == 14  # 2nd is length of title+body

    assert out['embedding'].iloc[0][1] == len('Title 1\nBody 1')


def test_embed_corpus_not_mutate_input():
    df = pd.DataFrame(
        {
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'body': ['Body 1', 'Body 2', 'Body 3'],
            'url': ['URL 1', 'URL 2', 'URL 3'],  # not required to embed
        }
    )

    embed_corpus(df, FakeEmbedder(), text_columns=['title', 'body'])
    # original df should not be mutated
    assert 'embedding' not in df.columns


class ScriptedEmbedder(VertexEmbedder):
    def __init__(self, batch_size: int = 2, failure_before_success: int = 0, error: Exception | None = None):
        self._batch_size = batch_size
        self.failures_left = failure_before_success
        self.attempts = 0
        self.error = error or ConnectionError('Simulated API failure 429')

    def _call_api(self, batch: list[str], task_type: str) -> list[list[float]]:  # noqa: ARG002
        self.attempts += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise self.error
        return [[float(len(t))] for t in batch]


def test_batching_covers_all_inputs_in_order():
    embedder = ScriptedEmbedder(batch_size=2)
    texts = ['text1', 'text2', 'text3', 'text4', 'text5']
    vectors = embedder.embed(texts)
    assert vectors.shape == (5, 1)
    assert [v[0] for v in vectors] == [len(t) for t in texts]


def test_empty_input_returns_empty_nocrash():
    embedder = ScriptedEmbedder()
    vectors = embedder.embed([])
    assert vectors.shape[0] == 0


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr('rag.embedding.time.sleep', lambda _s: None)  # avoid actual sleep

    embedder = ScriptedEmbedder(failure_before_success=2)
    vectors = embedder.embed(['hello'])
    assert embedder.attempts == 3  # 2 failures + 1 success
    assert vectors.shape == (1, 1)


def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr('rag.embedding.time.sleep', lambda _s: None)  # avoid actual sleep

    embedder = ScriptedEmbedder(failure_before_success=99)
    with pytest.raises(ConnectionError):
        embedder.embed(['hello'])
    assert embedder.attempts == 3


def test_non_retryable_error_fails_fast():
    # 400 = bug in my code not a transient failure. no retries, no sleep
    error = errors.APIError(400, {'error': {'message': 'bad request'}})
    embedder = ScriptedEmbedder(failure_before_success=99, error=error)
    with pytest.raises(errors.APIError):
        embedder.embed(['hello'])
    assert embedder.attempts == 1
