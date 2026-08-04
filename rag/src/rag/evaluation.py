import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from rag.answer import NO_COVERAGE_REPLY, Answer
from rag.config import Settings
from rag.models import ChunkHit, ChunksManifest
from rag.retrieval import Retriever, SearchMethod

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

    def format_line(self) -> str:
        recalls = '  '.join(f'recall@{k}={v:.2f}' for k, v in sorted(self.recall_at.items()))
        return f'n={self.n_queries}  {recalls}  MRR={self.mrr:.2f}'


class EvalRun(BaseModel):
    """Everything written to the timestamped run file: provenance + results."""

    created_at: datetime
    manifest: ChunksManifest
    method: SearchMethod
    rerank: bool
    k: int
    rrf_k: int
    rrf_vector_weight: float
    rrf_bm25_weight: float
    fts5_title_weight: float
    fts5_text_weight: float
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
    rerank: bool
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


def evaluate_query(
    query: EvalQuery,
    results: list[ChunkHit],
) -> QueryResult:
    """Score one query's retrieval results against expected URLs"""
    expected = {normalize_url(u) for u in query.expected_urls}

    rank: int | None = None
    for position, result in enumerate(results, start=1):
        if normalize_url(str(result.url)) in expected:
            rank = position
            break  # first hit -> all MRR and recall cares about

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
    return EvalSummary(
        n_queries=n,
        recall_at={k: sum(r.hit_at(k) for r in results) / n for k in ks},
        mrr=sum(r.reciprocal_rank for r in results) / n,
    )


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
    rerank: bool,
    k: int,
    results: list[QueryResult],
    settings: Settings,
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
        rerank=rerank,
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
    out_path = run_dir / f'{now:%Y-%m-%dT%H-%M-%S}_eval.json'
    out_path.write_text(run.model_dump_json(indent=2), encoding='utf-8')
    logger.info('wrote eval run to %s', out_path)
    return out_path, run


def write_answer_run(
    run_dir: Path,
    manifest: ChunksManifest,
    method: SearchMethod,
    rerank: bool,
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
        rerank=rerank,
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


def search_top_k_docs(
    retriever: Retriever, query: str, k: int, method: SearchMethod = 'hybrid', rerank: bool = False
) -> list[ChunkHit]:
    """Search and combine to k unique pages. If it falls short of k it widens the chunk fetch.

    One doc can occupy several of the top chunks in the retrieved list, so k * EVAL_OVERFETCH_FACTOR
    can still collapse to less than k unique docs. To ensure k results it doubles the fetch until either
    k docs are found, every chunk in the corpus has been retrieved, or the search returned fewer hits than
    asked for.
    """
    total_chunks = len(retriever)
    fetch_k = min(k * EVAL_OVERFETCH_FACTOR, total_chunks)
    while True:
        hits = retriever.search(query, k=fetch_k, method=method, rerank=rerank)
        collapsed = collapse_to_urls(hits, k)
        if len(collapsed) >= k or fetch_k >= total_chunks or len(hits) < fetch_k:
            return collapsed
        fetch_k = min(fetch_k * 2, total_chunks)


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
