import pytest

from rag.evaluation import EvalQuery, QueryResult, evaluate_query, summarize_results, validate_spans
from rag.models import ChunkHit
from rag.spans import SpanMatcher, score_ranking, tokenize

SPAN = 'You can choose to take a -1 penalty on all melee attack rolls to gain a +2 bonus on damage rolls.'


def _hit(chunk_id: str, text: str, url: str = 'https://example.com/alpha') -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split('#')[0],
        url=url,
        title='Alpha',
        heading_path=['Alpha'],
        text=text,
        category='feats',
        n_tokens=len(text.split()),
        full_article_length=500,
        score=0.9,
    )


# tokenize


def test_tokenize_lowercases_and_drops_markdown():
    assert tokenize('**Power** Attack') == ['power', 'attack']


def test_tokenize_keeps_modifier_signs():
    assert tokenize('a +2 bonus and a -1 penalty') == ['a', '+2', 'bonus', 'and', 'a', '-1', 'penalty']


def test_tokenize_collapses_whitespace_and_newlines():
    assert tokenize('one\n\n two \t three') == ['one', 'two', 'three']


# coverage


def test_exact_text_is_fully_covered():
    assert SpanMatcher([SPAN]).coverage(SPAN) == pytest.approx(1.0)


def test_coverage_survives_markdown_whitespace_and_case_differences():
    mangled = SPAN.replace('a -1', 'a **-1**').replace(' ', '\n').upper()
    assert SpanMatcher([SPAN]).coverage(mangled) == pytest.approx(1.0)


def test_unrelated_text_covers_nothing():
    unrelated = 'The druid gains an animal companion whose Hit Dice increase with her level.'
    assert SpanMatcher([SPAN]).coverage(unrelated) == pytest.approx(0.0)


def test_min_run_1_admits_stopword_coincidences_that_min_run_3_rejects():
    unrelated = 'A spellcaster can prepare a spell in a higher level slot on a given day.'
    assert SpanMatcher([SPAN], min_run=1).coverage(unrelated) > 0.0
    assert SpanMatcher([SPAN], min_run=3).coverage(unrelated) == pytest.approx(0.0)


def test_span_split_across_two_chunks_is_partial_on_each():
    head, tail = SPAN[:48], SPAN[48:]
    matcher = SpanMatcher([SPAN])
    assert 0.0 < matcher.coverage(head) < 1.0
    assert 0.0 < matcher.coverage(tail) < 1.0


def test_multiple_spans_share_one_coverage_denominator():
    matcher = SpanMatcher([SPAN, 'The bonus to damage is halved for off-hand weapons.'])
    covering_only_the_first = matcher.coverage(SPAN)
    assert 0.4 < covering_only_the_first < 0.9


def test_matcher_rejects_empty_span_list():
    with pytest.raises(ValueError, match='at least one span'):
        SpanMatcher([])


def test_matcher_rejects_span_that_tokenizes_to_nothing():
    with pytest.raises(ValueError, match='tokenize to nothing'):
        SpanMatcher(['***'])


# score_ranking


def test_straddling_span_is_covered_by_the_union_of_two_chunks():
    head, tail = SPAN[:48], SPAN[48:]
    score = score_ranking(SpanMatcher([SPAN]), [('a#000', head), ('a#001', tail)], ks=(1, 3))

    assert score.rank == 2  # not answerable until both chunks are in hand
    assert score.n_chunks_to_cover == 2
    assert score.coverage_at[1] < score.threshold
    assert score.coverage_at[3] >= score.threshold


def test_single_chunk_holding_the_whole_span_needs_one_chunk():
    score = score_ranking(SpanMatcher([SPAN]), [('a#000', SPAN)], ks=(1,))

    assert score.rank == 1
    assert score.n_chunks_to_cover == 1
    assert score.best_chunk_coverage == pytest.approx(1.0)
    assert score.best_chunk_id == 'a#000'
    assert score.reciprocal_rank == pytest.approx(1.0)


def test_rank_is_the_position_where_coverage_completes_not_where_it_starts():
    head, tail = SPAN[:48], SPAN[48:]
    ranked = [('a#000', head), ('b#000', 'unrelated filler text about oozes'), ('a#001', tail)]
    score = score_ranking(SpanMatcher([SPAN]), ranked, ks=(1, 3))

    assert score.rank == 3
    assert score.n_chunks_to_cover == 2  # the filler chunk contributed nothing


def test_span_never_covered_is_a_miss():
    score = score_ranking(SpanMatcher([SPAN]), [('b#000', 'nothing relevant here at all')], ks=(1,))

    assert score.rank is None
    assert score.is_miss
    assert score.reciprocal_rank == 0.0
    assert score.n_chunks_to_cover is None


def test_hit_at_respects_rank():
    score = score_ranking(SpanMatcher([SPAN]), [('b#000', 'filler'), ('a#000', SPAN)], ks=(1, 3))

    assert not score.hit_at(1)
    assert score.hit_at(3)


def test_ks_deeper_than_the_ranking_report_final_coverage():
    score = score_ranking(SpanMatcher([SPAN]), [('a#000', SPAN)], ks=(1, 50))

    assert score.coverage_at[50] == pytest.approx(1.0)


def test_best_chunk_coverage_ignores_rank():
    ranked = [('b#000', 'filler text'), ('a#000', SPAN)]
    score = score_ranking(SpanMatcher([SPAN]), ranked, ks=(1,))

    assert score.best_chunk_coverage == pytest.approx(1.0)
    assert score.coverage_at[1] == pytest.approx(0.0)


# EvalQuery


def test_query_accepts_spans():
    query = EvalQuery(query='power attack', type='exact_name', expected_urls=['https://x.com/a'], expected_spans=[SPAN])
    assert query.expected_spans == [SPAN]


def test_query_defaults_to_no_spans():
    query = EvalQuery(query='power attack', type='exact_name', expected_urls=['https://x.com/a'])
    assert query.expected_spans == []


def test_query_rejects_span_too_short_to_align():
    with pytest.raises(ValueError, match='too short'):
        EvalQuery(query='q', type='exact_name', expected_urls=['https://x.com/a'], expected_spans=['two words'])


# evaluate_query


def test_evaluate_query_scores_spans_off_the_chunk_ranking_not_the_collapsed_docs():
    query = EvalQuery(
        query='power attack', type='exact_name', expected_urls=['https://example.com/alpha'], expected_spans=[SPAN]
    )
    head, tail = SPAN[:48], SPAN[48:]
    chunks = [_hit('alpha#000', head), _hit('alpha#001', tail)]
    collapsed = [chunks[0]]  # what collapse_to_urls would leave: one chunk per doc

    result = evaluate_query(query, collapsed, chunk_ranking=chunks, ks=(1, 3))

    assert result.rank == 1  # URL metric is unchanged
    assert result.span is not None
    assert result.span.rank == 2  # span metric sees the second chunk the collapse dropped


def test_evaluate_query_without_chunk_ranking_leaves_span_unscored():
    query = EvalQuery(
        query='power attack', type='exact_name', expected_urls=['https://example.com/alpha'], expected_spans=[SPAN]
    )
    result = evaluate_query(query, [_hit('alpha#000', SPAN)])

    assert result.span is None


def test_evaluate_query_without_spans_leaves_span_unscored():
    query = EvalQuery(query='power attack', type='exact_name', expected_urls=['https://example.com/alpha'])
    chunks = [_hit('alpha#000', SPAN)]

    assert evaluate_query(query, chunks, chunk_ranking=chunks).span is None


# summarize_results


def _scored(rank: int | None) -> QueryResult:
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/alpha'], expected_spans=[SPAN])
    ranked = [('b#000', 'filler')] * ((rank or 1) - 1) + ([('a#000', SPAN)] if rank else [])
    return evaluate_query(query, [_hit('alpha#000', SPAN)], chunk_ranking=[_hit(c, t) for c, t in ranked], ks=(1, 3))


def test_summary_reports_span_metrics_over_the_scored_subset():
    unscored = evaluate_query(
        EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/alpha']),
        [_hit('alpha#000', SPAN)],
    )
    summary = summarize_results([_scored(1), _scored(3), unscored], ks=(1, 3))

    assert summary.n_queries == 3
    assert summary.n_with_spans == 2
    assert summary.span_recall_at[1] == pytest.approx(0.5)
    assert summary.span_recall_at[3] == pytest.approx(1.0)
    assert summary.span_mrr == pytest.approx((1.0 + 1 / 3) / 2)


def test_summary_span_fields_are_zero_when_no_query_carries_spans():
    result = evaluate_query(
        EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/alpha']),
        [_hit('alpha#000', SPAN)],
    )
    summary = summarize_results([result], ks=(1,))

    assert summary.n_with_spans == 0
    assert summary.span_recall_at == {}
    assert summary.mean_chunks_to_cover == 0.0


def test_summary_format_span_line_says_so_when_there_are_no_spans():
    result = evaluate_query(
        EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/alpha']),
        [_hit('alpha#000', SPAN)],
    )
    assert summarize_results([result], ks=(1,)).format_span_line() == 'no spans'


# validate_spans


def test_validate_passes_a_span_lifted_from_the_article_body():
    query = EvalQuery(
        query='power attack', type='exact_name', expected_urls=['https://example.com/alpha'], expected_spans=[SPAN]
    )
    bodies = {'https://example.com/alpha': f'# Power Attack\n\n{SPAN}\n\nNormal: you do not.'}

    [validation] = validate_spans([query], bodies)

    assert validation.ok
    assert validation.coverage == pytest.approx(1.0)
    assert validation.best_url == 'https://example.com/alpha'


def test_validate_fails_a_paraphrased_span():
    query = EvalQuery(
        query='power attack',
        type='exact_name',
        expected_urls=['https://example.com/alpha'],
        expected_spans=['Trade accuracy for damage by accepting an attack penalty in exchange.'],
    )
    bodies = {'https://example.com/alpha': SPAN}

    [validation] = validate_spans([query], bodies)

    assert not validation.ok
    assert validation.coverage < 0.9


def test_validate_fails_a_span_taken_from_the_wrong_page():
    query = EvalQuery(
        query='power attack', type='exact_name', expected_urls=['https://example.com/alpha'], expected_spans=[SPAN]
    )
    bodies = {'https://example.com/alpha': 'A completely different rule about grappling and pinning.'}

    [validation] = validate_spans([query], bodies)

    assert not validation.ok
    assert validation.best_url is None


def test_validate_matches_urls_after_normalization():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://Example.com/alpha/'], expected_spans=[SPAN])
    [validation] = validate_spans([query], {'https://example.com/alpha': SPAN})

    assert validation.ok


def test_validate_reports_nothing_for_queries_without_spans():
    query = EvalQuery(query='q', type='exact_name', expected_urls=['https://example.com/alpha'])
    assert validate_spans([query], {'https://example.com/alpha': SPAN}) == []


def test_tokenize_folds_typographic_dashes_to_ascii():
    assert tokenize('a –1 penalty') == tokenize('a -1 penalty') == ['a', '-1', 'penalty']  # noqa: RUF001


def test_coverage_matches_across_a_dash_style_difference():
    corpus_text = 'You can choose to take a –1 penalty on all melee attack rolls to gain a +2 bonus on damage rolls.'  # noqa: RUF001
    assert SpanMatcher([SPAN]).coverage(corpus_text) == pytest.approx(1.0)
