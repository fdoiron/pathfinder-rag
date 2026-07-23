import pytest
from typer.testing import CliRunner

from rag import cli
from rag.cli import app
from rag.models import ChunkHit
from rag.retrieval import ManifestMismatchError

runner = CliRunner()


class FakeEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass


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
    monkeypatch.setattr(cli, 'LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'nonexistent query'])
    assert result.exit_code == 0
    assert 'No results found.' in result.output


def test_search_prints_hits(monkeypatch):
    monkeypatch.setattr(cli, 'LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth'])
    assert result.exit_code == 0
    assert 'Alpha' in result.output


@pytest.mark.parametrize(
    'error',
    [
        FileNotFoundError('Chunks file not found: data/chunks.parquet'),
        ManifestMismatchError('embedding model mismatch'),
    ],
)
def test_search_load_retriever_failure_prints_clean_error(monkeypatch, error):
    monkeypatch.setattr(cli, 'LocalEmbedder', FakeEmbedder)

    def _raise(**kwargs):  # noqa: ARG001
        raise error

    monkeypatch.setattr(cli, 'load_retriever', _raise)
    result = runner.invoke(app, ['search', 'aboleth'])
    assert result.exit_code == 1
    assert f'Error: {error}' in result.output
