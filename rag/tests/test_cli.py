import pytest
from typer.testing import CliRunner

from rag import cli
from rag.answer import Answer, Citation, LLMUnavailableError
from rag.cli import app
from rag.models import ChunkHit
from rag.retrieval import ManifestMismatchError, OrphanChunksError

runner = CliRunner()


class FakeEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass


class UnloadableEmbedder:
    """Stands in for model weights that cannot be fetched"""

    def __init__(self, *_args, **_kwargs) -> None:
        raise OSError("Couldn't connect to https://huggingface.co")


class FakeRetriever:
    def __init__(self, hits: list[ChunkHit]) -> None:
        self._hits = hits

    def search(self, query: str, k: int, category: str | None = None) -> list[ChunkHit]:  # noqa: ARG002
        return self._hits


def _make_hit() -> ChunkHit:
    return ChunkHit(
        chunk_id='alpha#000',
        doc_id='alpha',
        url='https://example.com/alpha',
        title='Alpha',
        heading_path=['Alpha'],
        text='Text alpha',
        category='bestiary',
        n_tokens=10,
        score=0.9,
    )


def test_search_no_results_prints_message(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'nonexistent query'])
    assert result.exit_code == 0
    assert 'No results found.' in result.output


def test_search_prints_hits(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth'])
    assert result.exit_code == 0
    assert 'Alpha' in result.output


@pytest.mark.parametrize(
    'error',
    [
        FileNotFoundError('Chunks file not found: data/chunks.parquet'),
        ManifestMismatchError('embedding model mismatch'),
        OrphanChunksError('1 doc_id(s) have no match in data/corpus.parquet'),
    ],
)
def test_search_load_retriever_failure_prints_clean_error(monkeypatch, error):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)

    def _raise(**kwargs):  # noqa: ARG001
        raise error

    monkeypatch.setattr(cli, 'load_retriever', _raise)
    result = runner.invoke(app, ['search', 'aboleth'])
    assert result.exit_code == 1
    assert f'Error: {error}' in result.output


@pytest.mark.parametrize('command', [['search', 'aboleth'], ['ask', 'can I move and attack?']])
def test_embedder_load_failure_prints_clean_error(monkeypatch, command):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', UnloadableEmbedder)

    result = runner.invoke(app, command)

    assert result.exit_code == 1
    assert 'Error: Cannot load embedding model' in result.output
    assert 'huggingface.co' in result.output  # cause kept in the message


def test_ask_llm_unavailable_prints_clean_error(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise LLMUnavailableError('No LLM server reachable at http://localhost:11434/v1')

    monkeypatch.setattr(cli, 'answer_question', _raise)

    result = runner.invoke(app, ['ask', 'can I move and attack?'])

    assert result.exit_code == 1
    assert 'Error: No LLM server reachable' in result.output


def test_ask_prints_answer_and_citations(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    monkeypatch.setattr(
        cli,
        'answer_question',
        lambda *args, **kwargs: Answer(  # noqa: ARG005
            text='Yes, as a full-round action. [1]',
            citations=[Citation(n=1, title='Alpha', heading_path=['Alpha'], url='https://example.com/alpha')],
        ),
    )

    result = runner.invoke(app, ['ask', 'can I move and attack?'])

    assert result.exit_code == 0
    assert 'Yes, as a full-round action. [1]' in result.output
    assert 'https://example.com/alpha' in result.output
