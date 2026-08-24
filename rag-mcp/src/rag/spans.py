"""Chunk-independent scoring of an answer span against retrieved chunk text"""

import difflib
import re

from pydantic import BaseModel, ConfigDict

DEFAULT_COVERAGE_THRESHOLD = 0.9
DEFAULT_MIN_RUN = 3  # matched tokens per run. 3 drops stopword coincidences to zero on unrelated chunks


_NON_TOKEN = re.compile(r'[^a-z0-9+\-]+')

# The corpus writes penalties with typographic dashes ("a –1 penalty"). A span typed by hand has  # noqa: RUF003
# the ASCII hyphen which would tokenize to a different token and break the run it sits in.
_DASHES = str.maketrans({'‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '−': '-'})  # noqa: RUF001

# a (span index, token index) pair, so coverage unions across several spans in one set
Position = tuple[int, int]


def tokenize(text: str) -> list[str]:
    """Lowercase, fold unicode dashes to ASCII, split on everything that is not alphanumeric, '+' or '-'"""
    return _NON_TOKEN.sub(' ', text.lower().translate(_DASHES)).split()


def _matched_positions(span_tokens: list[str], chunk_tokens: list[str], min_run: int) -> set[int]:
    """Indices into span_tokens that chunk_tokens accounts for, counting runs of min_run or longer."""
    matcher = difflib.SequenceMatcher(a=span_tokens, b=chunk_tokens, autojunk=False)
    matched: set[int] = set()
    for span_start, _chunk_start, size in matcher.get_matching_blocks():
        if size >= min_run:
            matched.update(range(span_start, span_start + size))
    return matched


class SpanMatcher:
    """Tokenizes a query's spans once then scores arbitrary chunk text against them."""

    def __init__(self, spans: list[str], min_run: int = DEFAULT_MIN_RUN) -> None:
        if not spans:
            raise ValueError('SpanMatcher needs at least one span')
        self._span_tokens = [tokenize(span) for span in spans]
        empty = [i for i, tokens in enumerate(self._span_tokens) if not tokens]
        if empty:
            raise ValueError(f'span(s) {empty} tokenize to nothing')
        self.min_run = min_run
        self.n_tokens = sum(len(tokens) for tokens in self._span_tokens)

    def covered(self, text: str) -> set[Position]:
        """Which span positions this one text accounts for."""
        text_tokens = tokenize(text)
        return {
            (span_index, token_index)
            for span_index, span_tokens in enumerate(self._span_tokens)
            for token_index in _matched_positions(span_tokens, text_tokens, self.min_run)
        }

    def coverage(self, text: str) -> float:
        """Fraction of span tokens one text accounts for."""
        return len(self.covered(text)) / self.n_tokens


class SpanScore(BaseModel):
    """One query's span outcome over a ranked chunk list."""

    model_config = ConfigDict(extra='forbid')

    n_span_tokens: int
    threshold: float
    rank: int | None  # 1-based rank where cumulative union coverage first reaches threshold, None = never
    reciprocal_rank: float
    coverage_at: dict[int, float]  # k -> cumulative union coverage over ranks 1..k
    best_chunk_coverage: float  # most any single chunk covered, ranking aside
    best_chunk_id: str | None
    n_chunks_to_cover: int | None  # distinct chunks that contributed a token by `rank`; >1 means the span straddles

    @property
    def is_miss(self) -> bool:
        return self.rank is None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def score_ranking(
    matcher: SpanMatcher,
    ranked: list[tuple[str, str]],
    ks: tuple[int, ...],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> SpanScore:
    """Walk a ranked (chunk_id, text) list, accumulating union coverage"""
    covered: set[Position] = set()
    rank: int | None = None
    n_contributors = 0
    contributors_at_rank: int | None = None
    best_coverage = 0.0
    best_chunk_id: str | None = None
    coverage_at: dict[int, float] = {}

    for position, (chunk_id, text) in enumerate(ranked, start=1):
        chunk_covered = matcher.covered(text)

        chunk_coverage = len(chunk_covered) / matcher.n_tokens
        if chunk_coverage > best_coverage:
            best_coverage, best_chunk_id = chunk_coverage, chunk_id

        if chunk_covered - covered:
            n_contributors += 1
        covered |= chunk_covered

        if rank is None and len(covered) / matcher.n_tokens >= threshold:
            rank, contributors_at_rank = position, n_contributors

        if position in ks:
            coverage_at[position] = len(covered) / matcher.n_tokens

    # ks deeper than the ranking still get the final coverage: the list was exhausted, not truncated
    final = len(covered) / matcher.n_tokens
    for k in ks:
        coverage_at.setdefault(k, final)

    return SpanScore(
        n_span_tokens=matcher.n_tokens,
        threshold=threshold,
        rank=rank,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        coverage_at=coverage_at,
        best_chunk_coverage=best_coverage,
        best_chunk_id=best_chunk_id,
        n_chunks_to_cover=contributors_at_rank,
    )
