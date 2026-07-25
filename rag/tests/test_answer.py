from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import numpy as np
import pandas as pd
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError, OpenAI

from rag.answer import LLMUnavailableError, answer_question
from rag.config import Settings
from rag.models import ChunkHit, ChunksManifest
from rag.retrieval import Retriever


class FakeEmbedder:
    """Returns a fixed query vector regardless of input text."""

    def __init__(self, query_vec: list[float]):
        self._vec = np.array(query_vec, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([self._vec], dtype=np.float32)


class FakeChatClient:
    """Captures prompts, returns a canned completion."""

    def __init__(self, reply: str):
        self.prompts: list[str] = []
        self._reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:  # noqa: ARG002
        self.prompts.append(messages[-1]['content'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._reply))])


class FailingChatClient:
    """stands in for client with LLM server down or missing the model"""

    def __init__(self, error: Exception):
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:  # noqa: ARG002
        raise self._error


def _request() -> httpx.Request:
    return httpx.Request('POST', 'http://localhost:11434/v1/chat/completions')


def _status_error(error_cls: type[APIStatusError], status_code: int) -> APIStatusError:
    return error_cls('boom', response=httpx.Response(status_code, request=_request()), body=None)


def _make_manifest() -> ChunksManifest:
    return ChunksManifest(
        source_file='fake.parquet',
        source_sha256='abc123',
        n_articles=3,
        n_chunks=3,
        min_body_length=100,
        tokenizer_model='Qwen/Qwen3-Embedding-0.6B',
        max_tokens=450,
        overlap=50,
        parser_version='1',
        embedding_model='Qwen/Qwen3-Embedding-0.6B',
        embedding_dim=2,
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


def _make_retriever() -> Retriever:
    return Retriever(_make_chunks_df(), FakeEmbedder([1.0, 0.0]), _make_manifest())


class EmptyRetriever:
    """Stands in for a Retriever that finds nothing for the query."""

    def search(self, query: str, k: int, category: str | None = None) -> list[ChunkHit]:  # noqa: ARG002
        return []


def _make_settings(tmp_path: Path) -> Settings:
    prompt_path = tmp_path / 'ask.txt'
    prompt_path.write_text(
        'Excerpts:\n{excerpts}\n\nQuestion: {question}\n',
        encoding='utf-8',
    )
    return Settings(
        embedding_model='Qwen/Qwen3-Embedding-0.6B',
        embedding_dim=2,
        ask_prompt_path=prompt_path,
        ask_k=5,
    )


def test_prompt_contains_excerpts_and_question(tmp_path):
    retriever = _make_retriever()
    settings = _make_settings(tmp_path)
    client = FakeChatClient('Use Power Attack as a full-round option. [1][3]')

    answer_question('can I move and attack?', retriever, cast(OpenAI, client), settings)

    prompt = client.prompts[0]
    assert 'Text alpha' in prompt
    assert 'Text beta' in prompt
    assert 'Text gamma' in prompt
    assert '[1] Alpha' in prompt
    assert 'Question: can I move and attack?' in prompt


def test_citations_resolve_to_retrieved_hits(tmp_path):
    retriever = _make_retriever()
    settings = _make_settings(tmp_path)
    client = FakeChatClient('Use Power Attack as a full-round option. [1][3]')

    result = answer_question('can I move and attack?', retriever, cast(OpenAI, client), settings)

    assert [c.n for c in result.citations] == [1, 3]
    assert result.citations[0].url == 'https://example.com/alpha'
    assert result.citations[1].url == 'https://example.com/gamma'


def test_invented_citation_is_dropped_not_crashed(tmp_path):
    retriever = _make_retriever()
    settings = _make_settings(tmp_path)
    client = FakeChatClient('This cites a nonexistent excerpt. [9]')

    result = answer_question('anything', retriever, cast(OpenAI, client), settings)

    assert result.citations == []
    assert '[9]' in result.text


def test_no_coverage_reply_passes_through_with_no_citations(tmp_path):
    retriever = _make_retriever()
    settings = _make_settings(tmp_path)
    client = FakeChatClient("The retrieved excerpts don't cover this.")

    result = answer_question('anything', retriever, cast(OpenAI, client), settings)

    assert result.text == "The retrieved excerpts don't cover this."
    assert result.citations == []


def test_no_hits_returns_no_coverage_reply_without_calling_llm(tmp_path):
    settings = _make_settings(tmp_path)
    client = FakeChatClient('should never be called')

    result = answer_question('anything', cast(Retriever, EmptyRetriever()), cast(OpenAI, client), settings)

    assert result.text == "The retrieved excerpts don't cover this."
    assert result.citations == []
    assert client.prompts == []


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (APIConnectionError(request=_request()), 'No LLM server reachable'),  # server dead
        (APITimeoutError(request=_request()), 'did not answer within'),  # slow model load
        (_status_error(NotFoundError, 404), 'does not serve model'),  # model not pulled
        (_status_error(APIStatusError, 500), 'HTTP 500'),  # anything else from server
    ],
)
def test_llm_failure_becomes_llm_unavailable_error(tmp_path, error, expected):
    retriever = _make_retriever()
    settings = _make_settings(tmp_path)
    client = FailingChatClient(error)

    with pytest.raises(LLMUnavailableError, match=expected) as excinfo:
        answer_question('anything', retriever, cast(OpenAI, client), settings)

    assert excinfo.value.__cause__ is error
