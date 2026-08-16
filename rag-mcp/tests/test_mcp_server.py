import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest
from mcp.server.mcpserver import Context
from pydantic import SecretStr

from rag.config import Settings
from rag.mcp_server import (
    _STR,
    AppContext,
    ClassifiedToolError,
    SearchShedError,
    StaticTokenVerifier,
    _load_tool_strings,
    _ToolStrings,
    fetch_section,
    gpu_worker,
    rag_search,
)
from rag.models import Article, ChunkHit
from rag.retrieval import Retriever


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# helpers
def _hit(chunk_id: str, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id='skills__stealth',
        url=cast(Any, 'https://example.test/skills/stealth'),
        title='Stealth',
        heading_path=['Stealth'],
        text='You are trained at avoiding detection.',
        category='skills',
        n_tokens=12,
        full_article_length=4000,
        score=score,
    )


def _article(body_md: str) -> Article:
    return Article(
        doc_id='skills__stealth',
        url=cast(Any, 'https://example.test/skills/stealth'),
        title='Stealth',
        category='skills',
        breadcrumb=['Skills', 'Stealth'],
        body_md=body_md,
        n_chars=len(body_md),
    )


def _body(marker: str, offset: int, total: int) -> str:
    """body of exactly total chars with marker inserted at offset"""
    filler = 'x' * total
    return filler[:offset] + marker + filler[offset + len(marker) :]


class FakeRetriever:
    def __init__(
        self,
        hits: list[ChunkHit] | Exception | None = None,
        article: Article | Exception | None = None,
        chunk_text: str | None = None,
    ) -> None:
        self._hits = hits if hits is not None else []
        self._article = article
        self._chunk_text = chunk_text
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[ChunkHit]:
        self.searches.append(kwargs)
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits

    def get_article(self, chunk_id: str) -> Article | None:  # noqa: ARG002
        if isinstance(self._article, Exception):
            raise self._article
        return self._article

    def get_chunk_text(self, chunk_id: str) -> str | None:  # noqa: ARG002
        return self._chunk_text


@dataclass
class FakeRequestContext:
    lifespan_context: AppContext


@dataclass
class FakeContext:
    request_context: FakeRequestContext


def _app_ctx(retriever: FakeRetriever, *, maxsize: int = 8, accepting: bool = True) -> AppContext:
    return AppContext(
        settings=Settings(),
        retriever=cast(Retriever, retriever),
        gpu_queue=asyncio.Queue(maxsize=maxsize),
        worker_tasks=[],
        accepting=accepting,
    )


def _ctx(app_ctx: AppContext) -> Context[AppContext, Any]:
    return cast(Context[AppContext, Any], FakeContext(FakeRequestContext(app_ctx)))


async def _with_worker(app_ctx: AppContext) -> asyncio.Task[None]:
    task = asyncio.create_task(gpu_worker(0, app_ctx))
    await asyncio.sleep(0)  # let the worker reach its first queue.get()
    return task


# tool strings
def test_tool_strings_load_from_the_packaged_file() -> None:
    strings = _load_tool_strings()

    assert strings['rag_search.title'] == 'Search Pathfinder RAG'
    assert not any(key.startswith('#') for key in strings)  # comments are skipped. not parsed as keys
    assert all(value for value in strings.values())


def test_tool_strings_keep_colons_inside_the_value() -> None:
    strings = _load_tool_strings()

    assert strings['SearchResults.error_category'].startswith('Set only when results is empty')
    assert 'retryable: try again shortly' in strings['SearchResults.error_category']


def test_a_typoed_key_names_the_source_file() -> None:
    with pytest.raises(KeyError, match=r'mcp_tools\.txt'):
        _ToolStrings()['rag_search.titel']


# auth
@pytest.mark.anyio
async def test_the_configured_token_is_accepted() -> None:
    verifier = StaticTokenVerifier(SecretStr('s3cret'))

    access = await verifier.verify_token('s3cret')

    assert access is not None
    assert access.client_id == 'rag-search'


@pytest.mark.anyio
@pytest.mark.parametrize('presented', ['wrong', '', 's3cre', 's3cret '])
async def test_any_other_token_is_rejected(presented: str) -> None:
    verifier = StaticTokenVerifier(SecretStr('s3cret'))

    assert await verifier.verify_token(presented) is None


# rag_search
@pytest.mark.anyio
async def test_rag_search_ranks_and_maps_the_hits() -> None:
    retriever = FakeRetriever(hits=[_hit('skills__stealth#001', 0.9), _hit('skills__stealth#002', 0.4)])
    app_ctx = _app_ctx(retriever)
    worker = await _with_worker(app_ctx)

    try:
        results = await rag_search(query='how does stealth work', ctx=_ctx(app_ctx), k=5)
    finally:
        worker.cancel()

    assert results.error_category is None
    assert [r.rank for r in results.results] == [1, 2]
    assert [r.chunk_id for r in results.results] == ['skills__stealth#001', 'skills__stealth#002']
    assert results.results[0].body == 'You are trained at avoiding detection.'
    assert results.results[0].full_article_length == 4000
    assert retriever.searches[0]['category'] is None
    assert retriever.searches[0]['k'] == 5


@pytest.mark.anyio
async def test_rag_search_reports_an_empty_result_without_an_error_category() -> None:
    app_ctx = _app_ctx(FakeRetriever(hits=[]))
    worker = await _with_worker(app_ctx)

    try:
        results = await rag_search(query='how does stealth work', ctx=_ctx(app_ctx), category='skills')
    finally:
        worker.cancel()

    assert results.results == []
    assert results.message == _STR['rag_search.no_results_message']
    assert results.error_category is None  # no results is a valid answer


@pytest.mark.anyio
async def test_rag_search_refuses_new_work_while_shutting_down() -> None:
    app_ctx = _app_ctx(FakeRetriever(), accepting=False)

    results = await rag_search(query='how does stealth work', ctx=_ctx(app_ctx))

    assert results.results == []
    assert results.message == _STR['rag_search.shutdown_error']
    assert results.error_category == 'retryable'
    assert app_ctx.gpu_queue.qsize() == 0  # nothing was queued behind a closed door


@pytest.mark.anyio
async def test_rag_search_sheds_load_when_the_queue_is_full() -> None:
    app_ctx = _app_ctx(FakeRetriever(), maxsize=1)
    app_ctx.gpu_queue.put_nowait(cast(Any, object()))  # occupy the only slot

    results = await rag_search(query='how does stealth work', ctx=_ctx(app_ctx))

    assert results.results == []
    assert results.message == _STR['rag_search.busy_error']
    assert results.error_category == 'retryable'


@pytest.mark.anyio
async def test_rag_search_maps_a_drained_job_to_a_retry() -> None:
    app_ctx = _app_ctx(FakeRetriever())
    pending = asyncio.create_task(rag_search(query='how does stealth work', ctx=_ctx(app_ctx)))
    job = await app_ctx.gpu_queue.get()
    job.future.set_exception(SearchShedError())

    results = await pending

    assert results.message == _STR['rag_search.shutdown_error']
    assert results.error_category == 'retryable'


@pytest.mark.anyio
async def test_rag_search_reports_a_failed_search_as_fatal() -> None:
    app_ctx = _app_ctx(FakeRetriever(hits=RuntimeError('index is corrupt')))
    worker = await _with_worker(app_ctx)

    try:
        results = await rag_search(query='how does stealth work', ctx=_ctx(app_ctx))
    finally:
        worker.cancel()

    assert results.results == []
    assert results.message == 'Search failed: index is corrupt'
    assert results.error_category == 'fatal'


# fetch_section
@pytest.mark.anyio
async def test_fetch_section_returns_a_short_article_whole() -> None:
    body = 'Stealth rules.\n' * 20
    app_ctx = _app_ctx(FakeRetriever(article=_article(body), chunk_text='# Stealth\nStealth rules.'))

    window = await fetch_section(chunk_id='skills__stealth#001', ctx=_ctx(app_ctx))

    assert window.body_md == body
    assert (window.window_start, window.window_end) == (0, len(body))
    assert window.n_chars == len(body)
    assert window.title == 'Stealth'
    assert window.breadcrumb == ['Skills', 'Stealth']
    assert str(window.url) == 'https://example.test/skills/stealth'


@pytest.mark.anyio
@pytest.mark.parametrize(
    ('offset', 'expected_start', 'expected_end'),
    [
        (2000, 1510, 2510),  # chunk mid-article. Window is centred
        (10, 0, 1000),  # chunk near the top. window clamps to the start
        (3970, 3000, 4000),  # chunk near the bottom. window clamps to the end
    ],
)
async def test_fetch_section_windows_a_long_article_around_the_chunk(
    offset: int, expected_start: int, expected_end: int
) -> None:
    marker = 'M' * 20
    body = _body(marker, offset, 4000)
    app_ctx = _app_ctx(FakeRetriever(article=_article(body), chunk_text=f'# Stealth\n{marker}'))

    window = await fetch_section(chunk_id='skills__stealth#001', ctx=_ctx(app_ctx), max_chars=1000)

    assert (window.window_start, window.window_end) == (expected_start, expected_end)
    assert len(window.body_md) == 1000
    assert marker in window.body_md
    assert window.n_chars == 4000  # the full article length


@pytest.mark.anyio
async def test_fetch_section_falls_back_to_the_top_when_the_chunk_is_not_found_in_the_body() -> None:
    body = _body('M' * 20, 2000, 4000)
    app_ctx = _app_ctx(FakeRetriever(article=_article(body), chunk_text='# Stealth\nthis text is not in the body'))

    window = await fetch_section(chunk_id='skills__stealth#001', ctx=_ctx(app_ctx), max_chars=1000)

    assert (window.window_start, window.window_end) == (0, 1000)


@pytest.mark.anyio
async def test_fetch_section_tells_the_caller_to_rephrase_an_unknown_chunk_id() -> None:
    app_ctx = _app_ctx(FakeRetriever(article=None, chunk_text=None))

    with pytest.raises(ClassifiedToolError, match=r'\[rephrase\] Unknown chunk_id: bogus#001'):
        await fetch_section(chunk_id='bogus#001', ctx=_ctx(app_ctx))


@pytest.mark.anyio
async def test_fetch_section_reports_a_broken_retriever_as_fatal() -> None:
    app_ctx = _app_ctx(FakeRetriever(article=ValueError('get_article requires a Retriever built with docs')))

    with pytest.raises(ClassifiedToolError, match=r'\[fatal\].*server side bug'):
        await fetch_section(chunk_id='skills__stealth#001', ctx=_ctx(app_ctx))


# gpu_worker
@pytest.mark.anyio
async def test_the_worker_skips_a_job_the_caller_already_abandoned() -> None:
    retriever = FakeRetriever(hits=[_hit('skills__stealth#001', 0.9)])
    app_ctx = _app_ctx(retriever)
    pending = asyncio.create_task(rag_search(query='how does stealth work', ctx=_ctx(app_ctx)))
    await asyncio.sleep(0)  # let rag_search enqueue before anything drains the queue
    pending.cancel()
    worker = await _with_worker(app_ctx)

    try:
        await app_ctx.gpu_queue.join()
    finally:
        worker.cancel()

    assert retriever.searches == []  # no GPU time spent on a result nobody will read
