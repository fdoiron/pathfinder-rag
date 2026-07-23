import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rag.models import ChunkHit, ChunksManifest

logger = logging.getLogger(__name__)

RECALL_KS = (1, 3, 5)


# models


QueryType = Literal['exact_name', 'paraphrase', 'rules_reasoning']


class EvalQuery(BaseModel):
    """One line JSONL file"""

    query: str
    type: QueryType
    expected_urls: list[str] = Field(min_length=1)


class RetrievedItem(BaseModel):
    """Slim record for run file"""

    url: str
    title: str
    score: float


class QueryResult(BaseModel):
    """Outcome of evaluating single query"""

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
    k: int
    summary: EvalSummary
    by_type: dict[str, EvalSummary]
    by_category: dict[str, EvalSummary]
    results: list[QueryResult]


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
            RetrievedItem(
                url=str(r.url),
                title=r.title,
                score=r.score,
            )
            for r in results
        ],
        rank=rank,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
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


# run logging


def write_run(
    run_dir: Path,
    manifest: ChunksManifest,
    k: int,
    summary: EvalSummary,
    results: list[QueryResult],
) -> Path:
    """writes a timestamped run file"""
    summary = summarize_results(results)
    by_type = summarize_by(results, lambda r: r.type)
    by_category = summarize_by(results, lambda r: urlparse(r.expected_urls[0]).path.split('/')[1])
    now = datetime.now(UTC)
    run = EvalRun(
        created_at=now,
        manifest=manifest,
        k=k,
        summary=summary,
        by_type=by_type,
        by_category=by_category,
        results=results,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f'{now:%Y-%m-%dT%H-%M-%S}_eval.json'
    out_path.write_text(run.model_dump_json(indent=2), encoding='utf-8')
    logger.info('wrote eval run to %s', out_path)
    return out_path


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


def summarize_by(results: list[QueryResult], key: Callable[[QueryResult], str]) -> dict[str, EvalSummary]:
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {name: summarize_results(group) for name, group in sorted(groups.items())}
