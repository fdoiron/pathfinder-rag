import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, HttpUrl

from rag.config import Settings, get_settings
from rag.embedding import EmbedderUnavailableError, load_embedder
from rag.models import ChunkHit
from rag.reranking import RerankerUnavailableError, load_reranker
from rag.retrieval import ManifestMismatchError, OrphanChunksError, Retriever, StaleIndexError, load_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_WORKERS = 2


class SearchQuery(BaseModel):
    query: Annotated[str, Field(min_length=2, max_length=300, description='Keyword to search the RAG for')]
    k: Annotated[
        int, Field(ge=1, le=10, description='The Maximum number of chunks to return. May return less than k')
    ] = 5
    # TODO: this list is duplicated from the corpus's category set with no sync check. Validate against chunks.parquet
    # at startup, or move to a manifest file
    category: Annotated[
        Literal[
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
        | None,
        Field(description='Filter the topmost level'),
    ] = None


class FetchQuery(BaseModel):
    chunk_id: Annotated[str, Field(description='The chunk_id from search')]
    max_chars: Annotated[
        int,
        Field(
            ge=1000,
            le=60000,
            description='Upper bound on the returned body. A longer article is narrowed to a window around the chunk',
        ),
    ] = 12000


class ArticleWindow(BaseModel):
    """The source article for a chunk narrowed around it when the whole body would be too large."""

    doc_id: str
    title: str
    url: HttpUrl
    category: str
    breadcrumb: list[str]
    body_md: str = Field(description='Article body, or a window of it centred on the requested chunk')
    n_chars: int = Field(description='Length of the FULL article body, not of the returned excerpt')
    window_start: int = Field(description='Offset of the returned body within the full article (0 when whole)')
    window_end: int = Field(description='End offset of the returned body, exclusive. Equals n_chars when whole')


class ChunkSearchResult(BaseModel):
    rank: int = Field(ge=1, description='Rank position of this result (1-indexed)')
    score: float = Field(description='Score from Vector/BM25/Reranker')
    chunk_id: str = Field(description='Pass to fetch_section to retrieve the full source article')
    title: str = Field(description='Full title of the source article')
    url: HttpUrl = Field(description='URL of the source article')
    body: str = Field(description='Content of the retrieved chunk')
    article_n_chars: int = Field(description='Length in characters of the full source article (see fetch_section)')


class SearchResults(BaseModel):
    results: list[ChunkSearchResult]
    message: str | None = None


@dataclass
class SearchJob:
    query: str
    k: int
    category: str | None
    future: asyncio.Future[list[ChunkHit]] = field()


@dataclass
class AppContext:
    settings: Settings
    retriever: Retriever
    gpu_queue: asyncio.Queue[SearchJob]
    worker_tasks: list[asyncio.Task[None]]


async def gpu_worker(worker_id: int, ctx: AppContext) -> None:
    while True:
        job = await ctx.gpu_queue.get()
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
            if not job.future.cancelled():
                job.future.set_result(hits)
        except Exception as e:
            if not job.future.cancelled():
                job.future.set_exception(e)
        finally:
            ctx.gpu_queue.task_done()


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncGenerator[AppContext]:
    settings = get_settings()

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

    ctx = AppContext(settings=settings, retriever=retriever, gpu_queue=asyncio.Queue(maxsize=8), worker_tasks=[])
    ctx.worker_tasks = [asyncio.create_task(gpu_worker(i, ctx)) for i in range(N_WORKERS)]
    try:
        yield ctx
    finally:
        for task in ctx.worker_tasks:
            task.cancel()
        await asyncio.gather(*ctx.worker_tasks, return_exceptions=True)


mcp = MCPServer('rag-search', lifespan=app_lifespan)


@mcp.tool(
    title='Search Pathfinder RAG',
    annotations=ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    ),
)
async def rag_search(search_query: SearchQuery, ctx: Context[AppContext, Any]) -> SearchResults:
    """Search the RAG for pathfinder articles"""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    job = SearchJob(
        query=search_query.query,
        k=search_query.k,
        category=search_query.category,
        future=asyncio.get_running_loop().create_future(),
    )
    try:
        app_ctx.gpu_queue.put_nowait(job)
    except asyncio.QueueFull:
        raise ToolError('busy, retry in a few seconds') from None

    hits = await job.future

    if not hits:
        return SearchResults(results=[], message='No results found for this query/category combination.')

    return SearchResults(
        results=[
            ChunkSearchResult(
                rank=i + 1,
                score=h.score,
                chunk_id=h.chunk_id,
                title=h.title,
                url=h.url,
                body=h.text,
                article_n_chars=h.article_n_chars,
            )
            for i, h in enumerate(hits)
        ]
    )


@mcp.tool(
    title='Fetch the source article for a chunkID',
    annotations=ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    ),
)
async def fetch_section(fetch_query: FetchQuery, ctx: Context[AppContext, Any]) -> ArticleWindow:
    """Fetch the source article a chunk came from windowed around it if it is over max_chars"""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    retriever = app_ctx.retriever
    article = retriever.get_article(fetch_query.chunk_id)
    chunk_text = retriever.get_chunk_text(fetch_query.chunk_id)
    if article is None or chunk_text is None:
        raise ToolError(f'Unknown chunk_id: {fetch_query.chunk_id}')

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
