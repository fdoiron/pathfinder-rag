import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.answer import NO_COVERAGE_REPLY, Answer
from rag.config import Settings
from rag.models import ChunkHit, ChunksManifest
from rag.retrieval import Retriever, SearchMethod
from rag.spans import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_MIN_RUN,
    SpanMatcher,
    SpanScore,
    score_ranking,
)

logger = logging.getLogger(__name__)

RECALL_KS = (1, 3, 5, 20, 50)
EVAL_OVERFETCH_FACTOR = 5  # Headroom for initial chunks pere document when  collapsing chunk hits to unique pages


def url_category(url: str) -> str:
    """First path segment of a doc URL, ex: .../bestiary/aboleth' -> 'bestiary"""
    parts = urlparse(url).path.split('/')
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f'cannot derive category from url: {url!r}')
    return parts[1]


# models


QueryType = Literal['exact_name', 'paraphrase', 'rules_reasoning']


class EvalQuery(BaseModel):
    """One line JSONL file"""

    query: str
    type: QueryType
    expected_urls: list[str] = Field(min_length=1)
    expected_spans: list[str] = Field(default_factory=list)

    @field_validator('expected_spans')
    @classmethod
    def _spans_are_substantial(cls, spans: list[str]) -> list[str]:
        for span in spans:
            if len(span.split()) < DEFAULT_MIN_RUN:
                raise ValueError(f'span too short to align on: {span!r}')
        return spans

    @field_validator('expected_urls')
    @classmethod
    def _urls_have_category(cls, urls: list[str]) -> list[str]:
        for url in urls:
            url_category(url)  # raises ValueError if malformed
        return urls


class RetrievedItem(BaseModel):
    """Slim record for run file"""

    url: str
    title: str
    score: float
    chunk_id: str
    n_tokens: int


class QueryResult(BaseModel):
    """Outcome of evaluating single search query"""

    query: str
    type: QueryType
    expected_urls: list[str]
    retrieved_items: list[RetrievedItem]
    rank: int | None  # 1-based rank of 1st expected URL, None = miss
    reciprocal_rank: float
    span: SpanScore | None = None  # None = query carries no expected_spans

    @property
    def is_miss(self) -> bool:
        return self.rank is None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


class AnswerResult(BaseModel):
    """Outcome of evaluating a single answer"""

    # TODO: this is booleans only, can't audit offline.
    # Store answer.text and the cited URLs too
    query: str
    type: QueryType
    expected_urls: list[str]
    expected_url_retrieved: bool  # was the truth in what was fed to the LLM
    refused: bool  # text == NO_COVERAGE_REPLY
    citation_results: list[bool]  # per-citation: does it match an expected_url
    cited_correct_source: bool  # any(citation_results)
    invented_citation: bool  # bool(answer.invented_citations)


class EvalSummary(BaseModel):
    """Aggregate metrics over 1 eval"""

    n_queries: int
    recall_at: dict[int, float]  # k -> mean hit rate
    mrr: float
    n_with_spans: int = 0
    span_recall_at: dict[int, float] = Field(default_factory=dict)  # k -> mean span hit rate
    span_mrr: float = 0.0
    mean_coverage_at: dict[int, float] = Field(default_factory=dict)  # k -> mean union coverage
    mean_best_chunk_coverage: float = 0.0  # how much of the span is held by single best chunk
    mean_chunks_to_cover: float = 0.0  # >1 -> spans are straddling chunk boundaries

    def format_line(self) -> str:
        recalls = '  '.join(f'recall@{k}={v:.2f}' for k, v in sorted(self.recall_at.items()))
        return f'n={self.n_queries}  {recalls}  MRR={self.mrr:.2f}'

    def format_span_line(self) -> str:
        if not self.n_with_spans:
            return 'no spans'
        recalls = '  '.join(f'span_recall@{k}={v:.2f}' for k, v in sorted(self.span_recall_at.items()))
        return (
            f'n={self.n_with_spans}  {recalls}  span_MRR={self.span_mrr:.2f}  '
            f'best_chunk_cov={self.mean_best_chunk_coverage:.2f}  chunks_to_cover={self.mean_chunks_to_cover:.2f}'
        )


class EvalRun(BaseModel):
    """Everything written to the timestamped run file: provenance + results."""

    created_at: datetime
    manifest: ChunksManifest
    method: SearchMethod
    reranker_model: str | None
    reranker_dtype: str | None
    k: int
    rrf_k: int
    rrf_vector_weight: float
    rrf_bm25_weight: float
    fts5_title_weight: float
    fts5_text_weight: float
    hybrid_candidate_pool: int = 0  # 0 = run predates the field
    rerank_fetch_k: int | None = None
    rerank_prompt_path: str | None = None  # None = reranking.DEFAULT_INSTRUCTION
    span_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    span_min_run: int = DEFAULT_MIN_RUN
    summary: EvalSummary
    by_type: dict[str, EvalSummary]
    by_category: dict[str, EvalSummary]
    results: list[QueryResult]


class AnswerSummary(BaseModel):
    """Aggregate metrics over 1 answer eval"""

    n_queries: int
    retrieval_rate: float  # mean(expected_url_retrieved) -> was the truth fed to the LLM
    citation_rate: float  # mean(cited_correct_source)
    refusal_rate: float  # mean(refused)
    invented_citation_rate: float  # mean(invented_citation)

    def format_line(self) -> str:
        return (
            f'n={self.n_queries}  retrieved={self.retrieval_rate:.2f}  cited={self.citation_rate:.2f}  '
            f'refused={self.refusal_rate:.2f}  invented_citation={self.invented_citation_rate:.2f}'
        )


class AnswerEvalRun(BaseModel):
    """Everything written to the timestamped answer eval run file"""

    created_at: datetime
    manifest: ChunksManifest
    method: SearchMethod
    reranker_model: str | None
    reranker_dtype: str | None
    k: int
    rrf_k: int
    rrf_vector_weight: float
    rrf_bm25_weight: float
    fts5_title_weight: float
    fts5_text_weight: float
    summary: AnswerSummary
    by_type: dict[str, AnswerSummary]
    by_category: dict[str, AnswerSummary]
    results: list[AnswerResult]


# load source of truth


def load_queries(path: Path) -> list[EvalQuery]:
    """Parse  JSONL truth file, one EvalQuery per non-empty line"""
    queries: list[EvalQuery] = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            queries.append(EvalQuery.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f'{path}:{line_no}: invalid eval query: {exc}') from exc
    if not queries:
        raise ValueError(f'{path}: no queries found')
    logger.info('loaded %d eval queries from %s', len(queries), path)
    return queries


# calc metrics


def normalize_url(url: str) -> str:
    """lowercase scheme/host, no trailing slash"""
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip('/')
    return f'{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}'


class SpanValidation(BaseModel):
    """One span checked against the article body it claims to come from"""

    model_config = ConfigDict(extra='forbid')

    query: str
    span_index: int
    span: str
    coverage: float  # best coverage over the query's expected_urls
    best_url: str | None  # which expected_url's body accounted for it. None when no body was found
    ok: bool


def validate_spans(
    queries: list[EvalQuery],
    bodies_by_url: dict[str, str],
    min_coverage: float = DEFAULT_COVERAGE_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
) -> list[SpanValidation]:
    """Check every span actually occurs in one of its query's expected articles."""
    validations: list[SpanValidation] = []
    for query in queries:
        for index, span in enumerate(query.expected_spans):
            matcher = SpanMatcher([span], min_run=min_run)
            best_coverage, best_url = 0.0, None
            for url in query.expected_urls:
                body = bodies_by_url.get(normalize_url(url))
                if body is None:
                    continue
                coverage = matcher.coverage(body)
                if coverage > best_coverage:
                    best_coverage, best_url = coverage, url
            validations.append(
                SpanValidation(
                    query=query.query,
                    span_index=index,
                    span=span,
                    coverage=best_coverage,
                    best_url=best_url,
                    ok=best_coverage >= min_coverage,
                )
            )
    return validations


def evaluate_query(
    query: EvalQuery,
    results: list[ChunkHit],
    chunk_ranking: list[ChunkHit] | None = None,
    ks: tuple[int, ...] = RECALL_KS,
    span_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    span_min_run: int = DEFAULT_MIN_RUN,
) -> QueryResult:
    """Score one query's retrieval results against expected URLs, and its spans against chunk text."""
    expected = {normalize_url(u) for u in query.expected_urls}

    rank: int | None = None
    for position, result in enumerate(results, start=1):
        if normalize_url(str(result.url)) in expected:
            rank = position
            break  # first hit -> all MRR and recall cares about

    span: SpanScore | None = None
    if query.expected_spans and chunk_ranking is not None:
        span = score_ranking(
            SpanMatcher(query.expected_spans, min_run=span_min_run),
            [(hit.chunk_id, hit.text) for hit in chunk_ranking],
            ks=ks,
            threshold=span_threshold,
        )

    return QueryResult(
        query=query.query,
        type=query.type,
        expected_urls=query.expected_urls,
        retrieved_items=[
            RetrievedItem(url=str(r.url), title=r.title, score=r.score, chunk_id=r.chunk_id, n_tokens=r.n_tokens)
            for r in results
        ],
        rank=rank,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        span=span,
    )


def evaluate_answers(query: EvalQuery, answer: Answer, hits: list[ChunkHit]) -> AnswerResult:
    """Score one answer: was the truth retrieved, did it cite it, did it invent a citation, did it refuse"""
    expected = {normalize_url(u) for u in query.expected_urls}
    retrieval = evaluate_query(query, hits)  # reuse: rank is not None <=> truth was retrieved
    citation_results = [normalize_url(citation.url) in expected for citation in answer.citations]

    return AnswerResult(
        query=query.query,
        type=query.type,
        expected_urls=query.expected_urls,
        expected_url_retrieved=retrieval.rank is not None,
        refused=answer.text.strip() == NO_COVERAGE_REPLY,
        citation_results=citation_results,
        cited_correct_source=any(citation_results),
        invented_citation=bool(answer.invented_citations),
    )


def summarize_results(results: list[QueryResult], ks: tuple[int, ...] = RECALL_KS) -> EvalSummary:
    """Aggregate per query results into recall@k & MRR"""
    if not results:
        raise ValueError('cannot summarize an empty result list')

    n = len(results)
    scored = [r.span for r in results if r.span is not None]
    n_spans = len(scored)

    return EvalSummary(
        n_queries=n,
        recall_at={k: sum(r.hit_at(k) for r in results) / n for k in ks},
        mrr=sum(r.reciprocal_rank for r in results) / n,
        n_with_spans=n_spans,
        span_recall_at={k: sum(s.hit_at(k) for s in scored) / n_spans for k in ks} if n_spans else {},
        span_mrr=sum(s.reciprocal_rank for s in scored) / n_spans if n_spans else 0.0,
        mean_coverage_at=({k: sum(s.coverage_at.get(k, 0.0) for s in scored) / n_spans for k in ks} if n_spans else {}),
        mean_best_chunk_coverage=sum(s.best_chunk_coverage for s in scored) / n_spans if n_spans else 0.0,
        # averaged over covered spans only: an uncovered span has no chunk count to contribute
        mean_chunks_to_cover=_mean_chunks_to_cover(scored),
    )


def _mean_chunks_to_cover(scored: list[SpanScore]) -> float:
    counts = [s.n_chunks_to_cover for s in scored if s.n_chunks_to_cover is not None]
    return sum(counts) / len(counts) if counts else 0.0


def summarize_answers(results: list[AnswerResult]) -> AnswerSummary:
    """Aggregate results per answers into retrieval/citation/refusal/invented-citation rates"""
    if not results:
        raise ValueError('cannot summarize an empty result list')

    n = len(results)
    return AnswerSummary(
        n_queries=n,
        retrieval_rate=sum(r.expected_url_retrieved for r in results) / n,
        citation_rate=sum(r.cited_correct_source for r in results) / n,
        refusal_rate=sum(r.refused for r in results) / n,
        invented_citation_rate=sum(r.invented_citation for r in results) / n,
    )


# run logging


def write_run(
    run_dir: Path,
    manifest: ChunksManifest,
    method: SearchMethod,
    reranker_model: str | None,
    reranker_dtype: str | None,
    k: int,
    results: list[QueryResult],
    settings: Settings,
    fetch_k: int | None = None,
    span_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    span_min_run: int = DEFAULT_MIN_RUN,
) -> tuple[Path, EvalRun]:
    """Builds the EvalRun (summary + per-type/per-category breakdowns) and writes a timestamped run file"""
    ks = tuple(recall_k for recall_k in RECALL_KS if recall_k <= k)  # deeper ks were never searched, not measurable
    summary = summarize_results(results, ks=ks)
    by_type = summarize_by(results, lambda r: r.type, ks=ks)
    by_category = summarize_by(results, lambda r: url_category(r.expected_urls[0]), ks=ks)
    now = datetime.now(UTC)
    run = EvalRun(
        created_at=now,
        manifest=manifest,
        method=method,
        reranker_model=reranker_model,
        reranker_dtype=reranker_dtype,
        k=k,
        rrf_k=settings.rrf_k,
        rrf_vector_weight=settings.rrf_vector_weight,
        rrf_bm25_weight=settings.rrf_bm25_weight,
        fts5_title_weight=settings.fts5_title_weight,
        fts5_text_weight=settings.fts5_text_weight,
        hybrid_candidate_pool=settings.hybrid_candidate_pool,
        rerank_fetch_k=fetch_k,
        rerank_prompt_path=str(settings.rerank_prompt_path) if settings.rerank_prompt_path else None,
        span_threshold=span_threshold,
        span_min_run=span_min_run,
        summary=summary,
        by_type=by_type,
        by_category=by_category,
        results=results,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f'{now:%Y-%m-%dT%H-%M-%S}_eval.json'
    out_path.write_text(run.model_dump_json(indent=2), encoding='utf-8')
    logger.info('wrote eval run to %s', out_path)
    return out_path, run


def write_answer_run(
    run_dir: Path,
    manifest: ChunksManifest,
    method: SearchMethod,
    reranker_model: str | None,
    reranker_dtype: str | None,
    k: int,
    results: list[AnswerResult],
    settings: Settings,
) -> tuple[Path, AnswerEvalRun]:
    """Build AnswerEvalRun (summary + per-type and per-category breakdowns) and writes the run file"""
    summary = summarize_answers(results)
    by_type = summarize_answers_by(results, lambda r: r.type)
    by_category = summarize_answers_by(results, lambda r: url_category(r.expected_urls[0]))
    now = datetime.now(UTC)
    run = AnswerEvalRun(
        created_at=now,
        manifest=manifest,
        method=method,
        reranker_model=reranker_model,
        reranker_dtype=reranker_dtype,
        k=k,
        rrf_k=settings.rrf_k,
        rrf_vector_weight=settings.rrf_vector_weight,
        rrf_bm25_weight=settings.rrf_bm25_weight,
        fts5_title_weight=settings.fts5_title_weight,
        fts5_text_weight=settings.fts5_text_weight,
        summary=summary,
        by_type=by_type,
        by_category=by_category,
        results=results,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f'{now:%Y-%m-%dT%H-%M-%S}_answer_eval.json'
    out_path.write_text(run.model_dump_json(indent=2), encoding='utf-8')
    logger.info('wrote answer eval run to %s', out_path)
    return out_path, run


def collapse_to_urls(hits: list[ChunkHit], k: int) -> list[ChunkHit]:
    """turns Ranked chunk hits to ranked unique document hits with max of k

    A document's rank is its best chunk's rank. Following chunks of the same document are dropped
    """
    seen: set[str] = set()
    collapsed: list[ChunkHit] = []
    for hit in hits:
        key = normalize_url(str(hit.url))
        if key not in seen:
            seen.add(key)
            collapsed.append(hit)
    return collapsed[:k]


def search_docs_and_chunks(
    retriever: Retriever,
    query: str,
    k: int,
    method: SearchMethod = 'hybrid',
    rerank: bool = False,
    fetch_k: int | None = None,
) -> tuple[list[ChunkHit], list[ChunkHit]]:
    """Search once, return (k unique pages, the uncollapsed chunk ranking behind them).

    If the page list falls short of k it widens the chunk fetch. One doc can occupy several of the top
    chunks in the retrieved list, so k * EVAL_OVERFETCH_FACTOR can still collapse to less than k unique
    docs. To ensure k results it doubles the fetch until either k docs are found, every chunk in the
    corpus has been retrieved, or the search returned fewer hits than asked for.

    The raw hits are returned alongside because the span metrics score chunk granularity which is detroyed by collapsing
    Both metrics come from the same search so they always describe one ranking
    """
    total_chunks = len(retriever)
    search_k = min(k * EVAL_OVERFETCH_FACTOR, total_chunks)
    while True:
        hits = retriever.search(query, k=search_k, method=method, rerank=rerank, fetch_k=fetch_k)
        collapsed = collapse_to_urls(hits, k)
        if len(collapsed) >= k or search_k >= total_chunks or len(hits) < search_k:
            return collapsed, hits
        search_k = min(search_k * 2, total_chunks)


def search_top_k_docs(
    retriever: Retriever, query: str, k: int, method: SearchMethod = 'hybrid', rerank: bool = False
) -> list[ChunkHit]:
    """Search and combine to k unique pages and drop the chunk ranking"""
    return search_docs_and_chunks(retriever, query, k, method=method, rerank=rerank)[0]


def summarize_by(
    results: list[QueryResult], key: Callable[[QueryResult], str], ks: tuple[int, ...] = RECALL_KS
) -> dict[str, EvalSummary]:
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {name: summarize_results(group, ks=ks) for name, group in sorted(groups.items())}


def summarize_answers_by(results: list[AnswerResult], key: Callable[[AnswerResult], str]) -> dict[str, AnswerSummary]:
    groups: dict[str, list[AnswerResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {name: summarize_answers(group) for name, group in sorted(groups.items())}
