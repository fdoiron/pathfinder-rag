import sqlite3

import pandas as pd

from rag.lexical import _sanitize_match_query, build_fts5_index, search_fts5

# helpers


def _chunks_df(**overrides) -> pd.DataFrame:
    data = {
        'chunk_id': ['alpha#000', 'beta#000'],
        'heading_path': [['Alpha', 'Sub'], ['Beta']],
        'text': ['text alpha', 'text beta'],
        'category': ['bestiary', 'feats'],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _fts_rows(con: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    return con.execute('SELECT chunk_id, title, text, category FROM chunks_fts ORDER BY chunk_id').fetchall()


def _index(rows: list[dict[str, object]], fts5_tokenchar: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(':memory:')
    build_fts5_index(pd.DataFrame(rows), con, fts5_tokenchar=fts5_tokenchar)
    return con


# build_fts5_index


def test_indexes_every_row():
    con = sqlite3.connect(':memory:')
    build_fts5_index(_chunks_df(), con, fts5_tokenchar=False)
    assert con.execute('SELECT COUNT(*) FROM chunks_fts').fetchone()[0] == 2


def test_title_is_first_heading_path_element():
    con = sqlite3.connect(':memory:')
    build_fts5_index(_chunks_df(), con, fts5_tokenchar=False)
    rows = _fts_rows(con)
    assert rows == [
        ('alpha#000', 'Alpha', 'text alpha', 'bestiary'),
        ('beta#000', 'Beta', 'text beta', 'feats'),
    ]


def test_empty_heading_path_yields_empty_title():
    df = _chunks_df(heading_path=[[], None])
    con = sqlite3.connect(':memory:')
    build_fts5_index(df, con, fts5_tokenchar=False)
    titles = [row[1] for row in _fts_rows(con)]
    assert titles == ['', '']


def test_rebuild_replaces_rather_than_appends():
    con = sqlite3.connect(':memory:')
    build_fts5_index(_chunks_df(), con, fts5_tokenchar=False)
    smaller_df = _chunks_df(chunk_id=['gamma#000'], heading_path=[['Gamma']], text=['text gamma'], category=['spells'])
    build_fts5_index(smaller_df, con, fts5_tokenchar=False)
    assert _fts_rows(con) == [('gamma#000', 'Gamma', 'text gamma', 'spells')]


def test_tokenize_config_reflects_fts5_tokenchar_flag():
    con = sqlite3.connect(':memory:')
    build_fts5_index(_chunks_df(), con, fts5_tokenchar=True)
    create_sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    assert "tokenchars '-'" in create_sql


def test_tokenize_config_omits_tokenchars_when_flag_disabled():
    con = sqlite3.connect(':memory:')
    build_fts5_index(_chunks_df(), con, fts5_tokenchar=False)
    create_sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    assert 'tokenchars' not in create_sql


# search_fts5


def test_empty_query_returns_no_results():
    con = _index([{'chunk_id': 'a#000', 'heading_path': ['Alpha'], 'text': 'fireball spell', 'category': 'spells'}])
    results = search_fts5(con, '???', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert results == []


def test_returns_matching_chunk_id():
    con = _index([{'chunk_id': 'a#000', 'heading_path': ['Alpha'], 'text': 'fireball spell', 'category': 'spells'}])
    results = search_fts5(con, 'fireball', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert [chunk_id for chunk_id, _ in results] == ['a#000']


def test_non_matching_query_returns_no_results():
    con = _index([{'chunk_id': 'a#000', 'heading_path': ['Alpha'], 'text': 'fireball spell', 'category': 'spells'}])
    results = search_fts5(con, 'grapple', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert results == []


def test_results_ordered_best_first_by_more_negative_score():
    con = _index(
        [
            {
                'chunk_id': 'a#000',
                'heading_path': ['A'],
                'text': 'fireball fireball fireball filler filler filler',
                'category': 'spells',
            },
            {
                'chunk_id': 'b#000',
                'heading_path': ['B'],
                'text': 'fireball filler filler filler filler filler',
                'category': 'spells',
            },
        ]
    )
    results = search_fts5(con, 'fireball', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert [chunk_id for chunk_id, _ in results] == ['a#000', 'b#000']
    scores = [score for _, score in results]
    assert scores == sorted(scores)


def test_k_limits_number_of_results():
    rows: list[dict[str, object]] = [
        {'chunk_id': f'{c}#000', 'heading_path': [c], 'text': 'fireball spell', 'category': 'spells'}
        for c in ('a', 'b', 'c')
    ]
    con = _index(rows)
    results = search_fts5(con, 'fireball', k=2, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert len(results) == 2


def test_category_filters_out_other_categories():
    con = _index(
        [
            {'chunk_id': 'a#000', 'heading_path': ['A'], 'text': 'fireball spell', 'category': 'spells'},
            {'chunk_id': 'b#000', 'heading_path': ['B'], 'text': 'fireball trap', 'category': 'traps'},
        ]
    )
    results = search_fts5(
        con, 'fireball', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False, category='traps'
    )
    assert [chunk_id for chunk_id, _ in results] == ['b#000']


def test_title_weight_outranks_text_weight_when_higher():
    """Regression test for the bm25() column weight slot. The title_weight and text_weight must land on the
    title/text columns in order and not shift onto the wrong (UNINDEXED) column."""
    con = _index(
        [
            {
                'chunk_id': 'title_match#000',
                'heading_path': ['Fireball'],
                'text': 'deals damage in an area',
                'category': 'spells',
            },
            {
                'chunk_id': 'body_match#000',
                'heading_path': ['Misc'],
                'text': 'fireball deals damage in an area',
                'category': 'spells',
            },
        ]
    )
    title_heavy = search_fts5(con, 'fireball', k=5, title_weight=1000.0, text_weight=1.0, fts5_tokenchar=False)
    assert [chunk_id for chunk_id, _ in title_heavy] == ['title_match#000', 'body_match#000']

    text_heavy = search_fts5(con, 'fireball', k=5, title_weight=1.0, text_weight=1000.0, fts5_tokenchar=False)
    assert [chunk_id for chunk_id, _ in text_heavy] == ['body_match#000', 'title_match#000']


def test_hyphenated_term_only_matches_whole_token_when_tokenchar_enabled():
    con = _index(
        [{'chunk_id': 'a#000', 'heading_path': ['A'], 'text': 'aboleth-psionic attack', 'category': 'bestiary'}],
        fts5_tokenchar=True,
    )
    assert search_fts5(con, 'psionic', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=True) == []
    whole = search_fts5(con, 'aboleth-psionic', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=True)
    assert [chunk_id for chunk_id, _ in whole] == ['a#000']


def test_hyphenated_term_matches_either_half_when_tokenchar_disabled():
    con = _index(
        [{'chunk_id': 'a#000', 'heading_path': ['A'], 'text': 'aboleth-psionic attack', 'category': 'bestiary'}],
        fts5_tokenchar=False,
    )
    results = search_fts5(con, 'psionic', k=5, title_weight=1.0, text_weight=1.0, fts5_tokenchar=False)
    assert [chunk_id for chunk_id, _ in results] == ['a#000']


# _sanitize_match_query


def test_wraps_each_word_as_its_own_phrase():
    assert _sanitize_match_query('attack roll', fts5_tokenchar=False) == '"attack" "roll"'


def test_strips_trailing_punctuation():
    assert _sanitize_match_query('fireball?', fts5_tokenchar=False) == '"fireball"'


def test_strips_quotes_a_user_typed():
    assert _sanitize_match_query('the "fireball" spell', fts5_tokenchar=False) == '"the" "fireball" "spell"'


def test_empty_query_returns_empty_string():
    assert _sanitize_match_query('', fts5_tokenchar=False) == ''


def test_punctuation_only_query_returns_empty_string():
    assert _sanitize_match_query('???', fts5_tokenchar=False) == ''


def test_preserves_case():
    assert _sanitize_match_query('Aboleth', fts5_tokenchar=False) == '"Aboleth"'


def test_matches_unicode_word_chars():
    assert _sanitize_match_query('café', fts5_tokenchar=False) == '"café"'


def test_quoting_neutralizes_fts5_boolean_operators():
    assert _sanitize_match_query('fire AND ice', fts5_tokenchar=False) == '"fire" "AND" "ice"'


def test_hyphen_splits_words_when_tokenchar_disabled():
    assert _sanitize_match_query('aboleth-psionic', fts5_tokenchar=False) == '"aboleth" "psionic"'


def test_hyphen_kept_in_word_when_tokenchar_enabled():
    assert _sanitize_match_query('aboleth-psionic', fts5_tokenchar=True) == '"aboleth-psionic"'


def test_bare_hyphen_dropped_when_tokenchar_disabled():
    assert _sanitize_match_query('on - off', fts5_tokenchar=False) == '"on" "off"'
