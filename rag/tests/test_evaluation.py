from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.evaluation import (
    EvalQuery,
    EvalRun,
    QueryResult,
    evaluate_query,
    load_queries,
    normalize_url,
    summarize_results,
    write_run,
)
from rag.models import ChunkHit, ChunksManifest

# helpers


def make_result(url: str, title: str = 't', score: float = 0.9) -> ChunkHit:
    return ChunkHit(
        chunk_id='doc#000',
        doc_id='doc',
        url=url,
        title=title,
        heading_path=[],
        text='body',
        category='bestiary',
        n_tokens=4,
        score=score,
    )


def make_query_result(rank: int | None) -> QueryResult:
    rr = 0.0 if rank is None else 1.0 / rank
    return QueryResult(
        query='q',
        type='exact_name',
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
    summary = summarize_results(results)

    out_path = write_run(
        run_dir=tmp_path / 'runs',
        manifest=manifest,
        k=5,
        summary=summary,
        results=results,
    )

    assert out_path.exists()
    loaded = EvalRun.model_validate_json(out_path.read_text(encoding='utf-8'))
    assert loaded.k == 5
    assert loaded.summary.n_queries == summary.n_queries
    assert loaded.summary.mrr == pytest.approx(summary.mrr)
    assert loaded.manifest.n_articles == 10
    assert len(loaded.results) == 2
