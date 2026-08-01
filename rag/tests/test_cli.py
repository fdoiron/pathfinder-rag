import pytest
from typer.testing import CliRunner

from rag import cli
from rag.answer import Answer, Citation, LLMUnavailableError
from rag.cli import _apply_fts5_weight_overrides, app
from rag.config import Settings
from rag.models import ChunkHit
from rag.retrieval import ManifestMismatchError, OrphanChunksError, StaleIndexError

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

    def search(
        self,
        query: str,  # noqa: ARG002
        k: int,  # noqa: ARG002
        category: str | None = None,  # noqa: ARG002
        method: str = 'hybrid',  # noqa: ARG002
    ) -> list[ChunkHit]:
        return self._hits


class SpyRetriever:
    """Records the kwargs of the last search() call and returns fixed hits."""

    def __init__(self, hits: list[ChunkHit]) -> None:
        self._hits = hits
        self.last_call: dict[str, object] | None = None

    def search(self, query: str, k: int, category: str | None = None, method: str = 'hybrid') -> list[ChunkHit]:
        self.last_call = {'query': query, 'k': k, 'category': category, 'method': method}
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


def test_search_default_method_is_hybrid(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth'])
    assert spy.last_call is not None
    assert spy.last_call['method'] == 'hybrid'


@pytest.mark.parametrize('method', ['vector', 'bm25', 'hybrid'])
def test_search_method_flag_reaches_retriever(monkeypatch, method):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth', '--method', method])
    assert result.exit_code == 0
    assert spy.last_call is not None
    assert spy.last_call['method'] == method


def test_search_k_and_category_flags_reach_retriever(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth', '--k', '3', '--category', 'bestiary'])
    assert spy.last_call is not None
    assert spy.last_call['k'] == 3
    assert spy.last_call['category'] == 'bestiary'


def test_search_weight_flags_reach_load_retriever_settings(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    captured: dict[str, Settings] = {}

    def _capture(**kwargs) -> FakeRetriever:
        captured['settings'] = kwargs['settings']
        return FakeRetriever([_make_hit()])

    monkeypatch.setattr(cli, 'load_retriever', _capture)
    result = runner.invoke(app, ['search', 'aboleth', '--fts5-title-weight', '3.0', '--fts5-text-weight', '2.0'])
    assert result.exit_code == 0
    assert captured['settings'].fts5_title_weight == 3.0
    assert captured['settings'].fts5_text_weight == 2.0


def test_search_without_weight_flags_uses_settings_defaults(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    captured: dict[str, Settings] = {}

    def _capture(**kwargs) -> FakeRetriever:
        captured['settings'] = kwargs['settings']
        return FakeRetriever([_make_hit()])

    monkeypatch.setattr(cli, 'load_retriever', _capture)
    runner.invoke(app, ['search', 'aboleth'])
    assert captured['settings'].fts5_title_weight == Settings().fts5_title_weight
    assert captured['settings'].fts5_text_weight == Settings().fts5_text_weight


@pytest.mark.parametrize(
    'error',
    [
        FileNotFoundError('Chunks file not found: data/chunks.parquet'),
        ManifestMismatchError('embedding model mismatch'),
        OrphanChunksError('1 doc_id(s) have no match in data/corpus.parquet'),
        StaleIndexError('FTS5 index data/chunks.fts5.db does not match data/chunks.parquet'),
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


# _apply_fts5_weight_overrides


def test_no_overrides_returns_same_settings_object():
    settings = Settings()
    assert _apply_fts5_weight_overrides(settings, None, None) is settings


def test_title_weight_override_replaces_default_only():
    settings = Settings()
    result = _apply_fts5_weight_overrides(settings, 3.0, None)
    assert result.fts5_title_weight == 3.0
    assert result.fts5_text_weight == settings.fts5_text_weight


def test_text_weight_override_replaces_default_only():
    settings = Settings()
    result = _apply_fts5_weight_overrides(settings, None, 2.5)
    assert result.fts5_text_weight == 2.5
    assert result.fts5_title_weight == settings.fts5_title_weight


def test_both_overrides_applied_together():
    settings = Settings()
    result = _apply_fts5_weight_overrides(settings, 3.0, 2.5)
    assert result.fts5_title_weight == 3.0
    assert result.fts5_text_weight == 2.5


def test_override_does_not_mutate_original_settings():
    settings = Settings()
    original_title_weight = settings.fts5_title_weight
    _apply_fts5_weight_overrides(settings, 3.0, None)
    assert settings.fts5_title_weight == original_title_weight
