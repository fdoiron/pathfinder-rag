import numpy as np
import pandas as pd
import pytest
from google.genai import errors

from rag.config import Settings
from rag.corpus import embed_corpus
from rag.embedding import LocalEmbedder, VertexEmbedder


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


# ---------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------

QUERY_PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'


def _fake_st(*, prompts: dict[str, str], out_dim: int, calls: dict | None = None):
    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, model_kwargs=None):
            self.prompts = dict(prompts)
            if calls is not None:
                calls['init'] = {'model': model_name_or_path, 'model_kwargs': model_kwargs}

        def encode(self, texts, **kwargs):
            if calls is not None:
                calls['encode'] = {'texts': list(texts), 'kwargs': kwargs}
            # verify LocalEmbedder casts down to float32
            return np.ones((len(texts), out_dim), dtype=np.float64)

    return FakeSentenceTransformer


def _settings(dim: int = 4, batch_size: int = 2) -> Settings:
    return Settings(embedding_dim=dim, embedding_batch_size=batch_size)


def test_local_embedder_raises_without_query_prompt(monkeypatch):
    monkeypatch.setattr('rag.embedding.SentenceTransformer', _fake_st(prompts={}, out_dim=4))
    with pytest.raises(ValueError, match='query'):
        LocalEmbedder(_settings())


def test_query_prompt_exposes_model_prompt(monkeypatch):
    monkeypatch.setattr('rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4))
    embedder = LocalEmbedder(_settings())
    assert embedder.query_prompt == QUERY_PROMPT


def test_init_passes_model_and_dtype_to_sentence_transformer(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        'rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4, calls=calls)
    )
    LocalEmbedder(_settings())
    assert calls['init']['model'] == 'Qwen/Qwen3-Embedding-0.6B'
    assert 'torch_dtype' in calls['init']['model_kwargs']


def test_embed_query_uses_query_prompt_name(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        'rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4, calls=calls)
    )
    embedder = LocalEmbedder(_settings())
    embedder.embed(['what is a goblin?'], task_type='RETRIEVAL_QUERY')
    assert calls['encode']['kwargs']['prompt_name'] == 'query'


def test_embed_document_uses_no_prompt_name(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        'rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4, calls=calls)
    )
    embedder = LocalEmbedder(_settings())
    embedder.embed(['a goblin is a small humanoid'], task_type='RETRIEVAL_DOCUMENT')
    assert calls['encode']['kwargs']['prompt_name'] is None


def test_embed_normalizes_and_batches_from_settings(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        'rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4, calls=calls)
    )
    embedder = LocalEmbedder(_settings(batch_size=8))
    embedder.embed(['x', 'y'])
    assert calls['encode']['kwargs']['normalize_embeddings'] is True
    assert calls['encode']['kwargs']['batch_size'] == 8


def test_embed_returns_float32(monkeypatch):
    monkeypatch.setattr('rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4))
    embedder = LocalEmbedder(_settings(dim=4))
    vectors = embedder.embed(['a', 'b', 'c'])
    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32


def test_embed_empty_returns_empty_no_encode(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        'rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=4, calls=calls)
    )
    embedder = LocalEmbedder(_settings(dim=4))
    vectors = embedder.embed([])
    assert vectors.shape == (0, 4)
    assert vectors.dtype == np.float32
    assert 'encode' not in calls


def test_embed_dim_mismatch_raises(monkeypatch):
    monkeypatch.setattr('rag.embedding.SentenceTransformer', _fake_st(prompts={'query': QUERY_PROMPT}, out_dim=8))
    embedder = LocalEmbedder(_settings(dim=4))
    with pytest.raises(ValueError, match='embedding_dim'):
        embedder.embed(['a'])


@pytest.mark.gpu
def test_real_qwen3_embeds_finite_1024_dim_vectors():
    """Optional: load the real model. Run with uv run pytest -m gpu."""
    embedder = LocalEmbedder(Settings())
    vectors = embedder.embed(['An aboleth lurks beneath the lake.', 'Power Attack'])
    assert vectors.shape == (2, 1024)
    assert np.isfinite(vectors).all()
