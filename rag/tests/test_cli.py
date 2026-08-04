import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest
from typer.testing import CliRunner

from rag import cli
from rag.answer import NO_COVERAGE_REPLY, Answer, Citation, LLMUnavailableError
from rag.cli import _apply_fts5_weight_overrides, app
from rag.config import Settings
from rag.models import Article, Chunk, ChunkHit, ChunksManifest
from rag.retrieval import ManifestMismatchError, OrphanChunksError, StaleIndexError

runner = CliRunner()

_RETRIEVER_LOAD_ERRORS = [
    FileNotFoundError('Chunks file not found: data/chunks.parquet'),
    ManifestMismatchError('embedding model mismatch'),
    OrphanChunksError('1 doc_id(s) have no match in data/corpus.parquet'),
    StaleIndexError('FTS5 index data/chunks.fts5.db does not match data/chunks.parquet'),
]


def _make_manifest() -> ChunksManifest:
    return ChunksManifest(
        source_file='data/corpus.parquet',
        source_sha256='abc123',
        n_articles=1,
        n_chunks=1,
        min_body_length=100,
        tokenizer_model='Qwen/Qwen3-Embedding-0.6B',
        max_tokens=450,
        overlap=50,
        fts5_tokenchar=False,
        parser_version='1',
        embedding_model='Qwen/Qwen3-Embedding-0.6B',
        embedding_dim=2,
        embedding_dtype='float32',
        query_prompt='',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FakeReranker:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def torch_dtype(self) -> str:
        return 'torch.float32'


class FakeCorpusEmbedder:
    """Stands in for LocalEmbedder in build-corpus tests, which also reads torch_dtype/query_prompt."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        self.torch_dtype = 'float32'
        self.query_prompt = ''

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


class UnloadableEmbedder:
    """Stands in for model weights that cannot be fetched"""

    def __init__(self, *_args, **_kwargs) -> None:
        raise OSError("Couldn't connect to https://huggingface.co")


class UnloadableReranker:
    """Stands in for model weights that cannot be fetched"""

    def __init__(self, *_args, **_kwargs) -> None:
        raise OSError("Couldn't connect to https://huggingface.co")


class FakeRetriever:
    def __init__(self, hits: list[ChunkHit], manifest: ChunksManifest | None = None) -> None:
        self._hits = hits
        self.manifest = manifest or _make_manifest()

    def search(
        self,
        query: str,  # noqa: ARG002
        k: int,  # noqa: ARG002
        category: str | None = None,  # noqa: ARG002
        method: str = 'hybrid',  # noqa: ARG002
        rerank: bool = False,  # noqa: ARG002
        fetch_k: int | None = None,  # noqa: ARG002
    ) -> list[ChunkHit]:
        return self._hits


class SpyRetriever:
    """Records the kwargs of the last search() call and returns fixed hits."""

    def __init__(self, hits: list[ChunkHit]) -> None:
        self._hits = hits
        self.last_call: dict[str, object] | None = None

    def search(
        self,
        query: str,
        k: int,
        category: str | None = None,
        method: str = 'hybrid',
        rerank: bool = False,
        fetch_k: int | None = None,
    ) -> list[ChunkHit]:
        self.last_call = {
            'query': query,
            'k': k,
            'category': category,
            'method': method,
            'rerank': rerank,
            'fetch_k': fetch_k,
        }
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
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'nonexistent query'])
    assert result.exit_code == 0
    assert 'No results found.' in result.output


def test_search_prints_hits(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth'])
    assert result.exit_code == 0
    assert 'Alpha' in result.output


def test_search_default_method_is_hybrid(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth'])
    assert spy.last_call is not None
    assert spy.last_call['method'] == 'hybrid'


@pytest.mark.parametrize('method', ['vector', 'bm25', 'hybrid'])
def test_search_method_flag_reaches_retriever(monkeypatch, method):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth', '--method', method])
    assert result.exit_code == 0
    assert spy.last_call is not None
    assert spy.last_call['method'] == method


def test_search_k_and_category_flags_reach_retriever(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth', '--k', '3', '--category', 'bestiary'])
    assert spy.last_call is not None
    assert spy.last_call['k'] == 3
    assert spy.last_call['category'] == 'bestiary'


def test_search_default_rerank_is_true(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth'])
    assert spy.last_call is not None
    assert spy.last_call['rerank'] is True


def test_search_no_rerank_flag_reaches_retriever(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    spy = SpyRetriever([_make_hit()])
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: spy)  # noqa: ARG005
    runner.invoke(app, ['search', 'aboleth', '--no-rerank'])
    assert spy.last_call is not None
    assert spy.last_call['rerank'] is False


def test_search_no_rerank_flag_skips_reranker_load(monkeypatch):
    """--no-rerank must not even construct the reranker model."""
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', UnloadableReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    result = runner.invoke(app, ['search', 'aboleth', '--no-rerank'])
    assert result.exit_code == 0


def test_search_weight_flags_reach_load_retriever_settings(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
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
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    captured: dict[str, Settings] = {}

    def _capture(**kwargs) -> FakeRetriever:
        captured['settings'] = kwargs['settings']
        return FakeRetriever([_make_hit()])

    monkeypatch.setattr(cli, 'load_retriever', _capture)
    runner.invoke(app, ['search', 'aboleth'])
    assert captured['settings'].fts5_title_weight == Settings().fts5_title_weight
    assert captured['settings'].fts5_text_weight == Settings().fts5_text_weight


@pytest.mark.parametrize('flag', ['--fts5-title-weight', '--fts5-text-weight'])
@pytest.mark.parametrize('value', ['-5', '0'])
def test_search_rejects_non_positive_weight(flag, value):
    result = runner.invoke(app, ['search', 'aboleth', flag, value])
    assert result.exit_code == 2
    assert 'must be > 0' in result.output


@pytest.mark.parametrize('error', _RETRIEVER_LOAD_ERRORS)
def test_search_load_retriever_failure_prints_clean_error(monkeypatch, error):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)

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


@pytest.mark.parametrize('command', [['search', 'aboleth'], ['ask', 'can I move and attack?']])
def test_reranker_load_failure_prints_clean_error(monkeypatch, command):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', UnloadableReranker)

    result = runner.invoke(app, command)

    assert result.exit_code == 1
    assert 'Error: Cannot load reranking model' in result.output
    assert 'huggingface.co' in result.output  # cause kept in the message


def test_ask_llm_unavailable_prints_clean_error(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise LLMUnavailableError('No LLM server reachable at http://localhost:11434/v1')

    monkeypatch.setattr(cli, 'answer_question', _raise)

    result = runner.invoke(app, ['ask', 'can I move and attack?'])

    assert result.exit_code == 1
    assert 'Error: No LLM server reachable' in result.output


def test_ask_prints_answer_and_citations(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    monkeypatch.setattr(
        cli,
        'answer_question',
        lambda *args, **kwargs: (  # noqa: ARG005
            Answer(
                text='Yes, as a full-round action. [1]',
                citations=[Citation(n=1, title='Alpha', heading_path=['Alpha'], url='https://example.com/alpha')],
            ),
            [],
        ),
    )

    result = runner.invoke(app, ['ask', 'can I move and attack?'])

    assert result.exit_code == 0
    assert 'Yes, as a full-round action. [1]' in result.output
    assert 'https://example.com/alpha' in result.output


def test_ask_rerank_flag_reaches_answer_question(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    calls: list[bool] = []

    def _spy(question, retriever, client, settings, k=None, category=None, method='hybrid', rerank=False, fetch_k=None):  # noqa: ARG001
        calls.append(rerank)
        return Answer(text=NO_COVERAGE_REPLY, citations=[]), []

    monkeypatch.setattr(cli, 'answer_question', _spy)

    runner.invoke(app, ['ask', 'can I move and attack?'])
    runner.invoke(app, ['ask', 'can I move and attack?', '--no-rerank'])

    assert calls == [True, False]


def test_ask_no_rerank_flag_skips_reranker_load(monkeypatch):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', UnloadableReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda *args, **kwargs: FakeRetriever([_make_hit()]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    monkeypatch.setattr(
        cli,
        'answer_question',
        lambda *args, **kwargs: (Answer(text=NO_COVERAGE_REPLY, citations=[]), []),  # noqa: ARG005
    )

    result = runner.invoke(app, ['ask', 'can I move and attack?', '--no-rerank'])

    assert result.exit_code == 0


# evaluate


def _write_queries_file(tmp_path) -> str:
    content = (
        '{"query": "fireball", "type": "exact_name", "expected_urls": ["https://example.com/spells/fireball"]}\n'
        '{"query": "grapple", "type": "rules_reasoning", "expected_urls": ["https://example.com/combat/grapple"]}\n'
    )
    path = tmp_path / 'queries.jsonl'
    path.write_text(content, encoding='utf-8')
    return str(path)


def _fireball_hit() -> ChunkHit:
    return ChunkHit(
        chunk_id='fireball#000',
        doc_id='fireball',
        url='https://example.com/spells/fireball',
        title='Fireball',
        heading_path=['Fireball'],
        text='deals damage in an area',
        category='spells',
        n_tokens=6,
        score=0.95,
    )


def test_evaluate_writes_run_and_prints_summary(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005

    def _fake_search_top_k_docs(retriever, query, k, method='hybrid', rerank=False):  # noqa: ARG001
        return [_fireball_hit()] if query == 'fireball' else []

    monkeypatch.setattr(cli, 'search_top_k_docs', _fake_search_top_k_docs)
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    result = runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(run_dir)])

    assert result.exit_code == 0
    assert 'n=2' in result.output
    assert 'by type:' in result.output
    assert 'exact_name' in result.output
    assert 'rules_reasoning' in result.output
    assert 'by category:' in result.output
    assert 'spells' in result.output
    assert 'combat' in result.output
    assert '1 queries had no hits' in result.output
    assert 'grapple' in result.output
    assert list(run_dir.glob('*.json'))


@pytest.mark.parametrize('method', ['vector', 'bm25', 'hybrid'])
def test_evaluate_method_flag_reaches_search_top_k_docs(monkeypatch, tmp_path, method):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    calls: list[str] = []

    def _fake_search_top_k_docs(retriever, query, k, method='hybrid', rerank=False):  # noqa: ARG001
        calls.append(method)
        return []

    monkeypatch.setattr(cli, 'search_top_k_docs', _fake_search_top_k_docs)
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(tmp_path / 'runs'), '--method', method])

    assert result.exit_code == 0
    assert calls == [method, method]


def test_evaluate_default_method_is_hybrid_and_recorded_in_run(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'search_top_k_docs', lambda *a, **k: [])  # noqa: ARG005
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(run_dir)])

    [run_file] = run_dir.glob('*.json')
    assert '"method": "hybrid"' in run_file.read_text(encoding='utf-8')


def test_evaluate_default_rerank_is_true_and_recorded_in_run(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    calls: list[bool] = []

    def _fake_search_top_k_docs(retriever, query, k, method='hybrid', rerank=False):  # noqa: ARG001
        calls.append(rerank)
        return []

    monkeypatch.setattr(cli, 'search_top_k_docs', _fake_search_top_k_docs)
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(run_dir)])

    assert calls == [True, True]
    [run_file] = run_dir.glob('*.json')
    run_text = run_file.read_text(encoding='utf-8')
    assert '"reranker_model": "Qwen/Qwen3-Reranker-0.6B"' in run_text
    assert '"reranker_dtype": "torch.float32"' in run_text


def test_evaluate_no_rerank_flag_reaches_search_top_k_docs_and_skips_reranker_load(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', UnloadableReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    calls: list[bool] = []

    def _fake_search_top_k_docs(retriever, query, k, method='hybrid', rerank=False):  # noqa: ARG001
        calls.append(rerank)
        return []

    monkeypatch.setattr(cli, 'search_top_k_docs', _fake_search_top_k_docs)
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    result = runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(run_dir), '--no-rerank'])

    assert result.exit_code == 0
    assert calls == [False, False]
    [run_file] = run_dir.glob('*.json')
    run_text = run_file.read_text(encoding='utf-8')
    assert '"reranker_model": null' in run_text
    assert '"reranker_dtype": null' in run_text


def test_evaluate_weight_flags_reach_load_retriever_settings(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    captured: dict[str, Settings] = {}

    def _capture(**kwargs) -> FakeRetriever:
        captured['settings'] = kwargs['settings']
        return FakeRetriever([])

    monkeypatch.setattr(cli, 'load_retriever', _capture)
    monkeypatch.setattr(cli, 'search_top_k_docs', lambda *a, **k: [])  # noqa: ARG005
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(
        app,
        [
            'evaluate',
            queries_file,
            '--run-dir',
            str(tmp_path / 'runs'),
            '--fts5-title-weight',
            '3.0',
            '--fts5-text-weight',
            '2.0',
        ],
    )

    assert result.exit_code == 0
    assert captured['settings'].fts5_title_weight == 3.0
    assert captured['settings'].fts5_text_weight == 2.0


@pytest.mark.parametrize('flag', ['--fts5-title-weight', '--fts5-text-weight'])
@pytest.mark.parametrize('value', ['-5', '0'])
def test_evaluate_rejects_non_positive_weight(flag, value, tmp_path):
    queries_file = _write_queries_file(tmp_path)
    result = runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(tmp_path / 'runs'), flag, value])
    assert result.exit_code == 2
    assert 'must be > 0' in result.output


def test_evaluate_bad_queries_file_prints_clean_error(tmp_path):
    bad_file = tmp_path / 'bad.jsonl'
    bad_file.write_text('not valid json\n', encoding='utf-8')

    result = runner.invoke(app, ['evaluate', str(bad_file), '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 1
    assert 'Error loading queries:' in result.output


@pytest.mark.parametrize('error', _RETRIEVER_LOAD_ERRORS)
def test_evaluate_load_retriever_failure_prints_clean_error(monkeypatch, tmp_path, error):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)

    def _raise(**kwargs):  # noqa: ARG001
        raise error

    monkeypatch.setattr(cli, 'load_retriever', _raise)
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(app, ['evaluate', queries_file, '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 1
    assert f'Error: {error}' in result.output


# evaluate-answers


def _make_answer_question_stub(answers: dict[str, tuple[Answer, list[ChunkHit]]]):
    def _stub(
        question,
        retriever,  # noqa: ARG001
        client,  # noqa: ARG001
        settings,  # noqa: ARG001
        k=None,  # noqa: ARG001
        category=None,  # noqa: ARG001
        method='hybrid',  # noqa: ARG001
        rerank=False,  # noqa: ARG001
    ):
        return answers[question]

    return _stub


def test_evaluate_answers_writes_run_and_prints_summary(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    fireball_citation = Citation(
        n=1, title='Fireball', heading_path=['Fireball'], url='https://example.com/spells/fireball'
    )
    monkeypatch.setattr(
        cli,
        'answer_question',
        _make_answer_question_stub(
            {
                'fireball': (
                    Answer(text='Deals fire damage. [1]', citations=[fireball_citation]),
                    [_fireball_hit()],
                ),
                'grapple': (Answer(text=NO_COVERAGE_REPLY, citations=[]), []),
            }
        ),
    )
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    result = runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(run_dir)])

    assert result.exit_code == 0
    assert 'n=2' in result.output
    assert 'by type:' in result.output
    assert 'exact_name' in result.output
    assert 'rules_reasoning' in result.output
    assert 'by category:' in result.output
    assert list(run_dir.glob('*.json'))


def test_evaluate_answers_default_method_is_hybrid_and_recorded_in_run(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    monkeypatch.setattr(
        cli,
        'answer_question',
        _make_answer_question_stub(
            {
                'fireball': (Answer(text=NO_COVERAGE_REPLY, citations=[]), []),
                'grapple': (Answer(text=NO_COVERAGE_REPLY, citations=[]), []),
            }
        ),
    )
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(run_dir)])

    [run_file] = run_dir.glob('*.json')
    assert '"method": "hybrid"' in run_file.read_text(encoding='utf-8')


def test_evaluate_answers_default_rerank_is_true_and_recorded_in_run(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    calls: list[bool] = []

    def _spy(question, retriever, client, settings, k=None, category=None, method='hybrid', rerank=False, fetch_k=None):  # noqa: ARG001
        calls.append(rerank)
        return Answer(text=NO_COVERAGE_REPLY, citations=[]), []

    monkeypatch.setattr(cli, 'answer_question', _spy)
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(run_dir)])

    assert calls == [True, True]
    [run_file] = run_dir.glob('*.json')
    run_text = run_file.read_text(encoding='utf-8')
    assert '"reranker_model": "Qwen/Qwen3-Reranker-0.6B"' in run_text
    assert '"reranker_dtype": "torch.float32"' in run_text


def test_evaluate_answers_no_rerank_flag_skips_reranker_load(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', UnloadableReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    calls: list[bool] = []

    def _spy(question, retriever, client, settings, k=None, category=None, method='hybrid', rerank=False, fetch_k=None):  # noqa: ARG001
        calls.append(rerank)
        return Answer(text=NO_COVERAGE_REPLY, citations=[]), []

    monkeypatch.setattr(cli, 'answer_question', _spy)
    queries_file = _write_queries_file(tmp_path)
    run_dir = tmp_path / 'runs'

    result = runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(run_dir), '--no-rerank'])

    assert result.exit_code == 0
    assert calls == [False, False]
    [run_file] = run_dir.glob('*.json')
    run_text = run_file.read_text(encoding='utf-8')
    assert '"reranker_model": null' in run_text
    assert '"reranker_dtype": null' in run_text


def test_evaluate_answers_bad_queries_file_prints_clean_error(tmp_path):
    bad_file = tmp_path / 'bad.jsonl'
    bad_file.write_text('not valid json\n', encoding='utf-8')

    result = runner.invoke(app, ['evaluate-answers', str(bad_file), '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 1
    assert 'Error loading queries:' in result.output


@pytest.mark.parametrize('error', _RETRIEVER_LOAD_ERRORS)
def test_evaluate_answers_load_retriever_failure_prints_clean_error(monkeypatch, tmp_path, error):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)

    def _raise(**kwargs):  # noqa: ARG001
        raise error

    monkeypatch.setattr(cli, 'load_retriever', _raise)
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 1
    assert f'Error: {error}' in result.output


def test_evaluate_answers_llm_unavailable_prints_clean_error(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise LLMUnavailableError('No LLM server reachable at http://localhost:11434/v1')

    monkeypatch.setattr(cli, 'answer_question', _raise)
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 1
    assert 'Error: No LLM server reachable' in result.output


def test_evaluate_answers_reports_invented_citations(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeEmbedder)
    monkeypatch.setattr('rag.reranking.LocalReranker', FakeReranker)
    monkeypatch.setattr(cli, 'load_retriever', lambda **kwargs: FakeRetriever([]))  # noqa: ARG005
    monkeypatch.setattr(cli, 'make_llm_client', lambda settings: None)  # noqa: ARG005
    monkeypatch.setattr(
        cli,
        'answer_question',
        _make_answer_question_stub(
            {
                'fireball': (Answer(text='Made up. [9]', citations=[], invented_citations=[9]), []),
                'grapple': (Answer(text=NO_COVERAGE_REPLY, citations=[]), []),
            }
        ),
    )
    queries_file = _write_queries_file(tmp_path)

    result = runner.invoke(app, ['evaluate-answers', queries_file, '--run-dir', str(tmp_path / 'runs')])

    assert result.exit_code == 0
    assert 'cited something outside the retrieved excerpts' in result.output
    assert 'fireball' in result.output


# build-corpus


def _fake_article() -> Article:
    return Article(
        doc_id='fireball',
        url='https://example.com/spells/fireball',
        title='Fireball',
        category='spells',
        breadcrumb=['Spells', 'Fireball'],
        body_md='Deals damage in an area.',
        n_chars=25,
    )


def _fake_chunk() -> Chunk:
    return Chunk(
        chunk_id='fireball#000',
        doc_id='fireball',
        heading_path=['Fireball'],
        text='Deals damage in an area.',
        category='spells',
        n_tokens=6,
    )


def _build_corpus_settings(tmp_path, fts5_tokenchar: bool = False) -> Settings:
    return Settings(
        corpus_path=tmp_path / 'data' / 'corpus.parquet',
        chunks_path=tmp_path / 'data' / 'chunks.parquet',
        fts5_tokenchar=fts5_tokenchar,
    )


def test_build_corpus_writes_fts5_index(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeCorpusEmbedder)
    monkeypatch.setattr(cli, 'parse_corpus_dir', lambda html_dir, min_body_length: [_fake_article()])  # noqa: ARG005
    monkeypatch.setattr('rag.chunking.load_tokenizer', lambda model: object())  # noqa: ARG005
    monkeypatch.setattr(
        'rag.corpus.chunk_corpus',
        lambda articles, tokenizer, max_tokens, overlap: [_fake_chunk()],  # noqa: ARG005
    )
    settings = _build_corpus_settings(tmp_path)
    monkeypatch.setattr(cli, 'get_settings', lambda: settings)

    html_dir = tmp_path / 'html'
    html_dir.mkdir()

    result = runner.invoke(app, ['build-corpus', str(html_dir)])

    assert result.exit_code == 0
    fts_path = settings.chunks_path.with_suffix('.fts5.db')
    assert fts_path.exists()
    assert f'wrote fts5 index to {fts_path}' in result.output
    con = sqlite3.connect(fts_path)
    assert con.execute('SELECT COUNT(*) FROM chunks_fts').fetchone()[0] == 1


def test_build_corpus_fts5_tokenchar_setting_reaches_index(monkeypatch, tmp_path):
    monkeypatch.setattr('rag.embedding.LocalEmbedder', FakeCorpusEmbedder)
    monkeypatch.setattr(cli, 'parse_corpus_dir', lambda html_dir, min_body_length: [_fake_article()])  # noqa: ARG005
    monkeypatch.setattr('rag.chunking.load_tokenizer', lambda model: object())  # noqa: ARG005
    monkeypatch.setattr(
        'rag.corpus.chunk_corpus',
        lambda articles, tokenizer, max_tokens, overlap: [_fake_chunk()],  # noqa: ARG005
    )
    settings = _build_corpus_settings(tmp_path, fts5_tokenchar=True)
    monkeypatch.setattr(cli, 'get_settings', lambda: settings)

    html_dir = tmp_path / 'html'
    html_dir.mkdir()

    result = runner.invoke(app, ['build-corpus', str(html_dir)])

    assert result.exit_code == 0
    fts_path = settings.chunks_path.with_suffix('.fts5.db')
    con = sqlite3.connect(fts_path)
    create_sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    assert "tokenchars '-'" in create_sql


def test_build_corpus_no_articles_fails_clean_before_any_write(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, 'parse_corpus_dir', lambda html_dir, min_body_length: [])  # noqa: ARG005
    settings = _build_corpus_settings(tmp_path)
    monkeypatch.setattr(cli, 'get_settings', lambda: settings)

    html_dir = tmp_path / 'html'
    html_dir.mkdir()

    result = runner.invoke(app, ['build-corpus', str(html_dir)])

    assert result.exit_code == 1
    assert 'no articles parsed' in result.output
    assert not settings.corpus_path.exists()
    assert not settings.chunks_path.exists()


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
