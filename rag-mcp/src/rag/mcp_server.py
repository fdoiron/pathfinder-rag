import asyncio
import importlib.resources
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, get_args

from mcp.server._otel import OpenTelemetryMiddleware
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.metrics import CallbackOptions, Observation
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from rag.config import Settings, get_settings
from rag.embedding import EmbedderUnavailableError, load_embedder
from rag.models import ChunkHit
from rag.reranking import RerankerUnavailableError, load_reranker
from rag.retrieval import ManifestMismatchError, OrphanChunksError, Retriever, StaleIndexError, load_retriever
from rag.telemetry import configure_telemetry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
N_WORKERS = 1

# proxies until configure_telemetry() runs
tracer = trace.get_tracer('rag-search.worker')
meter = metrics.get_meter('rag-search')


class _ToolStrings(dict[str, str]):
    """Names the source file on a miss so a typoed key is not a KeyError at import."""

    def __missing__(self, key: str) -> str:
        raise KeyError(f'{key!r} is not defined in rag/prompts/mcp_tools.txt')


def _load_tool_strings() -> _ToolStrings:
    """Tool titles/descriptions/field descriptions/error messages from prompts/mcp_tools.txt."""
    text = importlib.resources.files('rag').joinpath('prompts', 'mcp_tools.txt').read_text(encoding='utf-8')
    strings = _ToolStrings()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition(': ')
        if not separator or not value:
            raise ValueError(f'mcp_tools.txt line {lineno}: expected "key: value", got {line!r}')
        strings[key] = value
    return strings


_STR = _load_tool_strings()


Category = Literal[
    'alignment-description',
    'basics-ability-scores',
    'bestiary',
    'classes',
    'equipment',
    'feats',
    'gamemastering',
    'magic-items',
    'magic',
    'races',
    'skills',
    'traits',
]
# duplicated from the corpus's own category set: app_lifespan checks it vs the loaded chunks
CATEGORIES: frozenset[str] = frozenset(get_args(Category))


class CategoryDriftError(RuntimeError):
    """The categories rag_search advertises no longer exist in the corpus it searches."""


# declared once and spread across both the flat tool signatures and the models they build
QueryArg = Annotated[str, Field(min_length=2, max_length=300, description=_STR['SearchQuery.query'])]
KArg = Annotated[int, Field(ge=5, le=10, description=_STR['SearchQuery.k'])]
CategoryArg = Annotated[Category | None, Field(description=_STR['SearchQuery.category'])]
ChunkIdArg = Annotated[str, Field(description=_STR['FetchQuery.chunk_id'])]
MaxCharsArg = Annotated[int, Field(ge=1000, le=60000, description=_STR['FetchQuery.max_chars'])]


class SearchQuery(BaseModel):
    query: QueryArg
    k: KArg = 5
    category: CategoryArg = None


class FetchQuery(BaseModel):
    chunk_id: ChunkIdArg
    max_chars: MaxCharsArg = 12000


class ArticleWindow(BaseModel):
    """The source article for a chunk narrowed around it when the whole body would be too large."""

    doc_id: str
    title: str
    url: HttpUrl
    category: str
    breadcrumb: list[str]
    body_md: str = Field(description=_STR['ArticleWindow.body_md'])
    n_chars: int = Field(description=_STR['ArticleWindow.n_chars'])
    window_start: int = Field(description=_STR['ArticleWindow.window_start'])
    window_end: int = Field(description=_STR['ArticleWindow.window_end'])


class ChunkSearchResult(BaseModel):
    rank: int = Field(ge=1, description=_STR['ChunkSearchResult.rank'])
    score: float = Field(description=_STR['ChunkSearchResult.score'])
    chunk_id: str = Field(description=_STR['ChunkSearchResult.chunk_id'])
    title: str = Field(description=_STR['ChunkSearchResult.title'])
    url: HttpUrl = Field(description=_STR['ChunkSearchResult.url'])
    body: str = Field(description=_STR['ChunkSearchResult.body'])
    full_article_length: int = Field(description=_STR['ChunkSearchResult.full_article_length'])


class SearchResults(BaseModel):
    results: list[ChunkSearchResult]
    message: str | None = None
    error_category: Annotated[
        Literal['retryable', 'rephrase', 'fatal'] | None, Field(description=_STR['SearchResults.error_category'])
    ] = None


class ClassifiedToolError(ToolError):
    def __init__(self, category: Literal['retryable', 'rephrase', 'fatal'], message: str):
        super().__init__(f'[{category}] {message}')


class StaticTokenVerifier(TokenVerifier):
    """Accepts a single shared bearer token compared in constant time."""

    def __init__(self, token: SecretStr) -> None:
        self._token = token.get_secret_value()

    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token.encode(), self._token.encode()):
            return None
        return AccessToken(token=token, client_id='rag-search', scopes=[])


class SearchShedError(Exception):
    """Set on a queued job's future when shutdown drains it unrun instead of ever being searched."""


@dataclass
class SearchJob:
    query: str
    k: int
    category: str | None
    future: asyncio.Future[list[ChunkHit]] = field()
    submitted_at: float = field(default_factory=time.monotonic)
    parent_ctx: otel_context.Context = field(default_factory=otel_context.get_current)


@dataclass
class AppContext:
    settings: Settings
    retriever: Retriever
    gpu_queue: asyncio.Queue[SearchJob]
    worker_tasks: list[asyncio.Task[None]]
    accepting: bool = True


def make_queue_depth_callback(ctx: AppContext) -> Callable[[CallbackOptions], Iterable[Observation]]:
    def callback(options: CallbackOptions) -> Iterable[Observation]:
        yield Observation(ctx.gpu_queue.qsize(), {})

    return callback


async def gpu_worker(worker_id: int, ctx: AppContext) -> None:
    while True:
        job = await ctx.gpu_queue.get()
        if job.future.cancelled():  # caller gave up while queued -> don't spend a worker on a result nobody will read
            logger.debug(f'worker {worker_id} skipping cancelled job')
            ctx.gpu_queue.task_done()
            continue
        token = otel_context.attach(job.parent_ctx)
        with tracer.start_as_current_span('gpu_worker.search') as span:
            span.set_attribute('queue.wait_ms', (time.monotonic() - job.submitted_at) * 1000)
            span.set_attribute('worker.id', worker_id)
            span.set_attribute('search.k', job.k)
            span.set_attribute('search.category', job.category or 'none')
            try:
                hits = await asyncio.to_thread(
                    ctx.retriever.search,
                    query=job.query,
                    k=job.k,
                    category=job.category,
                    method='hybrid',
                    rerank=True,
                    fetch_k=ctx.settings.rerank_fetch_k,
                )
                span.set_attribute('search.hit_count', len(hits))
                if not job.future.cancelled():
                    job.future.set_result(hits)
            except Exception as e:
                span.record_exception(e)
                logger.exception(f'worker {worker_id} search failed')
                if not job.future.cancelled():
                    job.future.set_exception(e)
            finally:
                otel_context.detach(token)
                ctx.gpu_queue.task_done()


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncGenerator[AppContext]:
    settings = get_settings()
    configure_telemetry(settings, 'rag-search', with_metrics=True)

    try:
        embedder = load_embedder(settings)
        reranker = load_reranker(settings)
        retriever = load_retriever(
            chunks_file=settings.chunks_path,
            embedder=embedder,
            reranker=reranker,
            settings=settings,
        )
    except (
        EmbedderUnavailableError,
        RerankerUnavailableError,
        FileNotFoundError,
        ManifestMismatchError,
        OrphanChunksError,
        StaleIndexError,
    ) as e:
        logger.error(f'Error: {e}')
        raise e

    if advertised_only := CATEGORIES - retriever.categories:
        raise CategoryDriftError(
            f'rag_search advertises categories absent from {settings.chunks_path}: {sorted(advertised_only)}'
        )
    if corpus_only := retriever.categories - CATEGORIES:
        logger.warning(f'corpus categories rag_search does not offer as a filter: {sorted(corpus_only)}')

    ctx = AppContext(settings=settings, retriever=retriever, gpu_queue=asyncio.Queue(maxsize=8), worker_tasks=[])
    ctx.worker_tasks = [asyncio.create_task(gpu_worker(i, ctx)) for i in range(N_WORKERS)]
    meter.create_observable_gauge(
        'rag_search.gpu_queue.depth',
        callbacks=[make_queue_depth_callback(ctx)],
        description='Current number of jobs waiting in the GPU search queue',
    )
    try:
        yield ctx
    finally:
        ctx.accepting = False  # refuse new searches then give the ones already queued a bounded window to finish
        try:
            await asyncio.wait_for(ctx.gpu_queue.join(), timeout=settings.mcp_drain_timeout)
        except TimeoutError:
            # qsize() would read 0 here while searches are still in flight. Report the window instead
            logger.warning(f'drain window of {settings.mcp_drain_timeout}s elapsed. Abandoning unfinished searches')
        # cancel() can't interrupt a search already inside to_thread -> only stops the worker waiting for it
        for task in ctx.worker_tasks:
            task.cancel()
        await asyncio.gather(*ctx.worker_tasks, return_exceptions=True)
        while True:
            try:
                job = ctx.gpu_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.set_exception(SearchShedError())


_auth_token = get_settings().mcp_auth_token
if _auth_token is None:
    logger.warning('RAG_MCP_AUTH_TOKEN unset: auth disabled. Every caller that can reach the port is authorized')

mcp = MCPServer(
    'rag-search',
    lifespan=app_lifespan,
    middleware=[OpenTelemetryMiddleware()],
    token_verifier=StaticTokenVerifier(_auth_token) if _auth_token else None,
    # nominal metadata: no authorization server exists at issuer_url but the SDK requires it
    auth=AuthSettings(
        issuer_url=get_settings().mcp_server_url,
        resource_server_url=get_settings().mcp_server_url,
    )
    if _auth_token
    else None,
)


@mcp.tool(
    title=_STR['rag_search.title'],
    description=_STR['rag_search.description'],
    # destructive_hint/idempotent_hint are meaningful only when read_only_hint is false
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
async def rag_search(
    query: QueryArg,
    ctx: Context[AppContext, Any],
    k: KArg = 5,
    category: CategoryArg = None,
) -> SearchResults:
    search_query = SearchQuery(query=query, k=k, category=category)
    app_ctx: AppContext = ctx.request_context.lifespan_context
    if not app_ctx.accepting:
        return SearchResults(results=[], message=_STR['rag_search.shutdown_error'], error_category='retryable')
    job = SearchJob(
        query=search_query.query,
        k=search_query.k,
        category=search_query.category,
        future=asyncio.get_running_loop().create_future(),
    )
    try:
        app_ctx.gpu_queue.put_nowait(job)
        logger.debug(f'job enqueued, depth={app_ctx.gpu_queue.qsize()}/{app_ctx.gpu_queue.maxsize}')
    except asyncio.QueueFull:
        logger.warning(f'gpu_queue full (maxsize={app_ctx.gpu_queue.maxsize}), rejecting job')
        return SearchResults(results=[], message=_STR['rag_search.busy_error'], error_category='retryable')

    try:
        hits = await job.future
    except SearchShedError:
        return SearchResults(results=[], message=_STR['rag_search.shutdown_error'], error_category='retryable')
    except Exception as e:
        logger.exception('search job failed')
        return SearchResults(results=[], message=f'Search failed: {e}', error_category='fatal')

    if not hits:
        return SearchResults(results=[], message=_STR['rag_search.no_results_message'])

    return SearchResults(
        results=[
            ChunkSearchResult(
                rank=i + 1,
                score=h.score,
                chunk_id=h.chunk_id,
                title=h.title,
                url=h.url,
                body=h.text,
                full_article_length=h.full_article_length,
            )
            for i, h in enumerate(hits)
        ]
    )


@mcp.tool(
    title=_STR['fetch_section.title'],
    description=_STR['fetch_section.description'],
    # destructive_hint/idempotent_hint are spec-meaningful only when read_only_hint is false
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
async def fetch_section(
    chunk_id: ChunkIdArg,
    ctx: Context[AppContext, Any],
    max_chars: MaxCharsArg = 12000,
) -> ArticleWindow:
    fetch_query = FetchQuery(chunk_id=chunk_id, max_chars=max_chars)
    app_ctx: AppContext = ctx.request_context.lifespan_context
    retriever = app_ctx.retriever
    try:
        article = retriever.get_article(fetch_query.chunk_id)
    except ValueError as e:
        logger.exception('get_article failed unexpectedly')
        raise ClassifiedToolError(
            'fatal', f'Internal error retrieving article. This is a server side bug not a bad request: {e}'
        ) from e
    chunk_text = retriever.get_chunk_text(fetch_query.chunk_id)
    if article is None or chunk_text is None:
        raise ClassifiedToolError(
            'rephrase', _STR['fetch_section.unknown_chunk_error'].format(chunk_id=fetch_query.chunk_id)
        )

    body = article.body_md
    max_chars = fetch_query.max_chars
    if len(body) <= max_chars:
        start, end = 0, len(body)
    else:
        snippet = max(chunk_text.splitlines()[1:], key=len, default='')  # [1:] skip heading prefix
        mid = max(body.find(snippet), 0) + len(snippet) // 2
        start = max(0, mid - max_chars // 2)
        end = min(len(body), start + max_chars)
        start = max(0, end - max_chars)

    return ArticleWindow(
        doc_id=article.doc_id,
        title=article.title,
        url=article.url,
        category=article.category,
        breadcrumb=article.breadcrumb,
        body_md=body[start:end],
        n_chars=article.n_chars,
        window_start=start,
        window_end=end,
    )


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
