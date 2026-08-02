import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag.evaluation import (
    EvalQuery,
    EvalRun,
    QueryResult,
    collapse_to_urls,
    evaluate_query,
    load_queries,
    normalize_url,
    search_top_k_docs,
    summarize_by,
    summarize_results,
    write_run,
)
from rag.lexical import build_fts5_index
from rag.models import ChunkHit, ChunksManifest
from rag.retrieval import Retriever

# helpers


def make_result(
    url: str, title: str = 't', score: float = 0.9, chunk_id: str = 'doc#000', n_tokens: int = 4
) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id='doc',
        url=url,
        title=title,
        heading_path=[],
        text='body',
        category='bestiary',
        n_tokens=n_tokens,
        score=score,
    )


def make_query_result(rank: int | None, qtype: str = 'exact_name') -> QueryResult:
    rr = 0.0 if rank is None else 1.0 / rank
    return QueryResult(
        query='q',
        type=qtype,
        expected_urls=['https://example.com/a'],
        retrieved_items=[],
        rank=rank,
        reciprocal_rank=rr,
    )


def make_manifest() -> ChunksManifest:
    return ChunksManifest(
        source_file='data/corpus.parquet',
        source_sha256='abc123',
        n_articles=10,
        n_chunks=15,
        min_body_length=100,
        tokenizer_model='Qwen/Qwen3-Embedding-0.6B',
        max_tokens=450,
        overlap=50,
        fts5_tokenchar=False,
        parser_version='1',
        embedding_model='Qwen/Qwen3-Embedding-0.6B',
        embedding_dim=1024,
        embedding_dtype='float32',
        query_prompt='',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# norm url


def test_normalize_strips_trailing_slash():
    assert normalize_url('https://example.com/article/') == 'https://example.com/article'


def test_normalize_lowercases_host_not_path():
    assert normalize_url('HTTPS://EXAMPLE.COM') == 'https://example.com'


def test_normalize_handles_whitespace():
    assert normalize_url(' https://x.ca/a ') == 'https://x.ca/a'


# evaluate_query tests


def test_hit_at_rank_1():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/article'])
    results = [make_result('https://example.com/article')]
    qr = evaluate_query(query, results)
    assert qr.rank == 1
    assert qr.reciprocal_rank == 1.0
    assert qr.hit_at(1) is True


def test_hit_at_rank_4():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/d'])
    results = [
        make_result('https://example.com/a'),
        make_result('https://example.com/b'),
        make_result('https://example.com/c'),
        make_result('https://example.com/d'),
        make_result('https://example.com/e'),
    ]
    qr = evaluate_query(query, results)
    assert qr.rank == 4
    assert qr.reciprocal_rank == 0.25
    assert qr.hit_at(3) is False
    assert qr.hit_at(5) is True


def test_miss():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/missing'])
    results = [make_result('https://example.com/other')]
    qr = evaluate_query(query, results)
    assert qr.rank is None
    assert qr.reciprocal_rank == 0.0
    assert qr.is_miss is True


def test_second_expected_url_counts():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/x', 'https://example.com/b'])
    results = [
        make_result('https://example.com/a'),
        make_result('https://example.com/b'),
    ]
    qr = evaluate_query(query, results)
    assert qr.rank == 2


def test_first_hit_wins_when_both_present():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/b', 'https://example.com/d'])
    results = [
        make_result('https://example.com/a'),
        make_result('https://example.com/b'),
        make_result('https://example.com/c'),
        make_result('https://example.com/d'),
    ]
    qr = evaluate_query(query, results)
    assert qr.rank == 2


def test_url_normalization_applied_in_matching():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/article/'])
    results = [make_result('https://example.com/article')]
    qr = evaluate_query(query, results)
    assert qr.rank == 1


def test_empty_results_is_miss():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/article'])
    qr = evaluate_query(query, [])
    assert qr.rank is None
    assert qr.reciprocal_rank == 0.0
    assert qr.is_miss is True


def test_retrieved_preserved_in_order():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/a'])
    urls = ['https://example.com/a', 'https://example.com/b', 'https://example.com/c']
    results = [make_result(u) for u in urls]
    qr = evaluate_query(query, results)
    assert [item.url for item in qr.retrieved_items] == urls


def test_chunk_id_and_n_tokens_preserved_per_item():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/a'])
    results = [
        make_result('https://example.com/a', chunk_id='a#000', n_tokens=10),
        make_result('https://example.com/b', chunk_id='b#002', n_tokens=25),
    ]
    qr = evaluate_query(query, results)
    assert [(item.chunk_id, item.n_tokens) for item in qr.retrieved_items] == [('a#000', 10), ('b#002', 25)]


# collapse_to_urls


def test_collapse_dedupes_by_url_keeping_first():
    hit1 = make_result('https://example.com/a', title='a1')
    hit2 = make_result('https://example.com/b', title='b')
    hit3 = make_result('https://example.com/a', title='a2')
    collapsed = collapse_to_urls([hit1, hit2, hit3], k=5)
    assert collapsed == [hit1, hit2]


def test_collapse_applies_k_after_dedupe_not_before():
    hit_a1 = make_result('https://example.com/a')
    hit_a2 = make_result('https://example.com/a')
    hit_b = make_result('https://example.com/b')
    collapsed = collapse_to_urls([hit_a1, hit_a2, hit_b], k=2)
    assert [str(h.url) for h in collapsed] == ['https://example.com/a', 'https://example.com/b']


# search_top_k_docs


class _FakeEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = np.array(vec, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT') -> np.ndarray:  # noqa: ARG002
        return np.array([self._vec] * len(texts), dtype=np.float32)


def _make_retriever(n: int) -> Retriever:
    chunks_df = pd.DataFrame(
        {
            'chunk_id': [f'doc{i}#000' for i in range(n)],
            'doc_id': [f'doc{i}' for i in range(n)],
            'title': [f'Title {i}' for i in range(n)],
            'url': [f'https://example.com/{i}' for i in range(n)],
            'heading_path': [[f'Title {i}'] for i in range(n)],
            'text': [f'body text {i}' for i in range(n)],
            'category': ['bestiary'] * n,
            'n_tokens': [5] * n,
            'embedding': [np.array([1.0, 0.0], dtype=np.float32) for _ in range(n)],
        }
    )
    fts_con = sqlite3.connect(':memory:')
    build_fts5_index(chunks_df, fts_con, fts5_tokenchar=False)
    return Retriever(chunks_df, _FakeEmbedder([1.0, 0.0]), make_manifest(), fts_con=fts_con)


def test_search_top_k_docs_stops_widening_when_bm25_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BM25 query matching 0 chunks returns 0 chunks at any fetch_k. The loop must not keep rerunning that same
    exhausted MATCH up to the full corpus size before giving up."""
    retriever = _make_retriever(n=100)
    calls: list[int] = []
    original_search = retriever.search

    def counting_search(query: str, k: int, category: str | None = None, method: str = 'hybrid'):
        calls.append(k)
        return original_search(query, k, category=category, method=method)  # type: ignore[arg-type]

    monkeypatch.setattr(retriever, 'search', counting_search)

    result = search_top_k_docs(retriever, 'zzznomatchzzz', k=5, method='bm25')

    assert result == []
    assert calls == [25]


def test_search_top_k_docs_stops_widening_when_exhausted_after_a_widen(monkeypatch: pytest.MonkeyPatch) -> None:
    """25 chunks match the query term but all collapse to the same doc and no other chunk in the corpus
    matches. The loop must widen once (fetch_k 25 -> 50), see the second fetch still return fewer hits than
    requested, and stop there. It must not keep doubling fetch_k up to the full n=100 corpus."""
    n_dup = 25
    n_filler = 75
    total = n_dup + n_filler
    chunks_df = pd.DataFrame(
        {
            'chunk_id': [f'dup#{i:03d}' for i in range(n_dup)] + [f'filler{i}#000' for i in range(n_filler)],
            'doc_id': ['dup'] * n_dup + [f'filler{i}' for i in range(n_filler)],
            'title': ['Dup'] * n_dup + [f'Filler {i}' for i in range(n_filler)],
            'url': ['https://example.com/dup'] * n_dup + [f'https://example.com/filler{i}' for i in range(n_filler)],
            'heading_path': [['Dup']] * n_dup + [[f'Filler {i}'] for i in range(n_filler)],
            'text': ['zzzmatchzzz'] * n_dup + ['unrelated text'] * n_filler,
            'category': ['bestiary'] * total,
            'n_tokens': [5] * total,
            'embedding': [np.array([1.0, 0.0], dtype=np.float32) for _ in range(total)],
        }
    )
    fts_con = sqlite3.connect(':memory:')
    build_fts5_index(chunks_df, fts_con, fts5_tokenchar=False)
    retriever = Retriever(chunks_df, _FakeEmbedder([1.0, 0.0]), make_manifest(), fts_con=fts_con)

    calls: list[int] = []
    original_search = retriever.search

    def counting_search(query: str, k: int, category: str | None = None, method: str = 'hybrid'):
        calls.append(k)
        return original_search(query, k, category=category, method=method)  # type: ignore[arg-type]

    monkeypatch.setattr(retriever, 'search', counting_search)

    result = search_top_k_docs(retriever, 'zzzmatchzzz', k=5, method='bm25')

    assert calls == [25, 50]
    assert [r.doc_id for r in result] == ['dup']


def test_search_top_k_docs_widens_on_duplicate_doc_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preexisting widening trigger (collapsed < k after deduplication) still works through search_top_k_docs:
    the top 10 vector hits all collapse to one doc so it must widen to fetch 20 before it can find a 2nd."""
    n_dup = 10
    n_unique = 10
    total = n_dup + n_unique
    scores = np.linspace(1.0, 0.5, total)
    chunks_df = pd.DataFrame(
        {
            'chunk_id': [f'dup#{i:03d}' for i in range(n_dup)] + [f'doc{i}#000' for i in range(n_unique)],
            'doc_id': ['dup'] * n_dup + [f'doc{i}' for i in range(n_unique)],
            'title': ['Dup'] * n_dup + [f'Title {i}' for i in range(n_unique)],
            'url': ['https://example.com/dup'] * n_dup + [f'https://example.com/doc{i}' for i in range(n_unique)],
            'heading_path': [['Dup']] * n_dup + [[f'Title {i}'] for i in range(n_unique)],
            'text': ['text'] * total,
            'category': ['bestiary'] * total,
            'n_tokens': [5] * total,
            'embedding': [np.array([s, (1 - s**2) ** 0.5], dtype=np.float32) for s in scores],
        }
    )
    retriever = Retriever(chunks_df, _FakeEmbedder([1.0, 0.0]), make_manifest())

    calls: list[int] = []
    original_search = retriever.search

    def counting_search(query: str, k: int, category: str | None = None, method: str = 'hybrid'):
        calls.append(k)
        return original_search(query, k, category=category, method=method)  # type: ignore[arg-type]

    monkeypatch.setattr(retriever, 'search', counting_search)

    result = search_top_k_docs(retriever, 'anything', k=2, method='vector')

    assert calls == [10, 20]
    assert [r.doc_id for r in result] == ['dup', 'doc0']


# @summarize results


def test_summary_known_mix():
    # ranks: 1, 2, None -> rr: 1.0, 0.5, 0.0 -> mrr = 0.5
    results = [make_query_result(1), make_query_result(2), make_query_result(None)]
    summary = summarize_results(results)
    assert summary.recall_at[1] == pytest.approx(1 / 3)
    assert summary.recall_at[3] == pytest.approx(2 / 3)
    assert summary.recall_at[5] == pytest.approx(2 / 3)
    assert summary.mrr == pytest.approx(0.5)


def test_summary_all_hits_at_1():
    results = [make_query_result(1), make_query_result(1)]
    summary = summarize_results(results)
    assert summary.recall_at[1] == pytest.approx(1.0)
    assert summary.recall_at[3] == pytest.approx(1.0)
    assert summary.recall_at[5] == pytest.approx(1.0)
    assert summary.mrr == pytest.approx(1.0)


def test_summary_empty_raises():
    with pytest.raises(ValueError, match='cannot summarize an empty result list'):
        summarize_results([])


# summarize_by


def test_summarize_by_groups_and_computes_metrics_per_group():
    # exact_name: ranks 1, None -> mrr 0.5, recall@1 0.5
    # paraphrase: ranks 2, 1    -> mrr 0.75, recall@1 0.5
    results = [
        make_query_result(1, qtype='exact_name'),
        make_query_result(None, qtype='exact_name'),
        make_query_result(2, qtype='paraphrase'),
        make_query_result(1, qtype='paraphrase'),
    ]
    by_type = summarize_by(results, lambda r: r.type)
    assert set(by_type) == {'exact_name', 'paraphrase'}
    assert by_type['exact_name'].mrr == pytest.approx(0.5)
    assert by_type['exact_name'].recall_at[1] == pytest.approx(0.5)
    assert by_type['paraphrase'].mrr == pytest.approx(0.75)
    assert by_type['paraphrase'].recall_at[1] == pytest.approx(0.5)


# load queries


def test_load_valid_jsonl(tmp_path: Path):
    content = (
        '{"query": "q1", "type": "exact_name", "expected_urls": ["https://example.com/a"]}\n'
        '{"query": "q2", "type": "paraphrase", "expected_urls": ["https://example.com/b"]}\n'
        '\n'
    )
    f = tmp_path / 'queries.jsonl'
    f.write_text(content, encoding='utf-8')
    queries = load_queries(f)
    assert len(queries) == 2
    assert queries[0].query == 'q1'
    assert queries[0].expected_urls == ['https://example.com/a']
    assert queries[1].query == 'q2'


def test_load_bad_line_raises_with_line_number(tmp_path: Path):
    content = '{"query": "q1", "type": "exact_name", "expected_urls": ["https://example.com/a"]}\nnot valid json\n'
    f = tmp_path / 'queries.jsonl'
    f.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='2'):
        load_queries(f)


def test_load_empty_file_raises(tmp_path: Path):
    f = tmp_path / 'queries.jsonl'
    f.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='no queries found'):
        load_queries(f)


def test_load_missing_expected_urls_raises(tmp_path: Path):
    content = '{"query": "q1", "type": "exact_name", "expected_urls": []}\n'
    f = tmp_path / 'queries.jsonl'
    f.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='invalid eval query'):
        load_queries(f)


# write_run


def test_write_run_creates_readable_file(tmp_path: Path):
    manifest = make_manifest()
    results = [make_query_result(1), make_query_result(None)]

    out_path, run = write_run(
        run_dir=tmp_path / 'runs',
        manifest=manifest,
        method='hybrid',
        k=5,
        results=results,
    )

    assert out_path.exists()
    assert run.k == 5
    assert run.summary.n_queries == 2
    assert run.summary.mrr == pytest.approx(0.5)
    assert 'exact_name' in run.by_type

    loaded = EvalRun.model_validate_json(out_path.read_text(encoding='utf-8'))
    assert loaded.k == 5
    assert loaded.summary.n_queries == run.summary.n_queries
    assert loaded.summary.mrr == pytest.approx(run.summary.mrr)
    assert loaded.manifest.n_articles == 10
    assert len(loaded.results) == 2
