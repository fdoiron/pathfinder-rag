import functools
from pathlib import Path

import pytest

from rag.chunking import (
    _HEADING_RE,
    Section,
    _pack_lines,
    _pack_sentences,
    _pack_table_rows,
    _split_body,
    _split_by_markdown_heading,
    _split_sentences,
    chunk_article,
)
from rag.models import Article

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


class _WordTokenizer:
    """Stand-in tokenizer. One token per whitespaced elimited word"""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[str]]:
        ids = text.split()
        if add_special_tokens:
            ids = [*ids, '<eos>']  # emulates Qwen3-Embedding appending EOS. _calc_tokens must not count EOS
        return {'input_ids': ids}

    def decode(self, ids: list[str]) -> str:
        return ' '.join(ids)


_tok = _WordTokenizer()


def _make_article(body_md: str) -> Article:
    return Article(
        doc_id='bestiary__x__y',
        url='https://www.d20pfsrd.com/bestiary/x/y',
        title='XY Title',
        category='bestiary',
        breadcrumb=['Home', 'X'],
        body_md=body_md,
        n_chars=len(body_md),
    )


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_DIR = FIXTURES_DIR / 'goldens'
GOLDEN_SLUGS = sorted(p.stem.removesuffix('.golden') for p in GOLDEN_DIR.glob('*.golden.md'))


@functools.cache
def _split_golden(slug: str) -> tuple[str, tuple[Section, ...]]:
    """Read and split golden once and shared over the golden tests"""
    body = (GOLDEN_DIR / f'{slug}.golden.md').read_text(encoding='utf-8')
    sections = tuple(_split_by_markdown_heading(_make_article(body)))
    return body, sections


# ---------------------------------------------------------------
# _split_by_markdown_heading
# ---------------------------------------------------------------


def test_split_no_headings_returns_single_section_with_empty_path():
    article = _make_article('Plain text with no headings.')
    assert _split_by_markdown_heading(article) == [Section((), 'Plain text with no headings.')]


def test_split_no_headings_strips_surrounding_whitespace():
    article = _make_article('  Plain text with no headings.  ')
    assert _split_by_markdown_heading(article) == [Section((), 'Plain text with no headings.')]


def test_split_empty_article_returns_single_empty_section():
    assert _split_by_markdown_heading(_make_article('')) == [Section((), '')]


def test_split_text_then_heading_keeps_preamble_as_same_section():
    article = _make_article('Preamble text.\n\n# First Heading\n\nBody.')
    sections = _split_by_markdown_heading(article)
    assert sections[0] == Section((), 'Preamble text.')
    assert sections[1] == Section(('First Heading',), 'Body.')


def test_split_heading_at_start_has_no_preamble_section():
    article = _make_article('# Only Heading\n\nBody.')
    sections = _split_by_markdown_heading(article)
    assert sections == [Section(('Only Heading',), 'Body.')]


def test_split_nested():
    body = '# H1\n\na\n\n## H2\n\nb\n\n### H3\n\nc\n\n## H2b\n\nd'
    sections = _split_by_markdown_heading(_make_article(body))
    assert [s.heading_path for s in sections] == [
        ('H1',),
        ('H1', 'H2'),
        ('H1', 'H2', 'H3'),
        ('H1', 'H2b'),
    ]


def test_split_level_skip_h1_to_h3():
    body = '# H1\n\na\n\n### H3\n\nb'
    sections = _split_by_markdown_heading(_make_article(body))
    assert [s.heading_path for s in sections] == [('H1',), ('H1', 'H3')]


def test_split_sibling_heading_does_not_accumulate():
    body = '# One\n\na\n\n# Two\n\nb\n\n# Three\n\nc'
    sections = _split_by_markdown_heading(_make_article(body))
    assert [s.heading_path for s in sections] == [('One',), ('Two',), ('Three',)]


def test_split_heading_at_eof_has_empty_body():
    body = 'Intro\n\n# Trailing Heading'
    sections = _split_by_markdown_heading(_make_article(body))
    assert sections[-1] == Section(('Trailing Heading',), '')


def test_split_back_to_back_headings_have_empty_body_between():
    body = '# First\n\n# Second\n\nBody.'
    sections = _split_by_markdown_heading(_make_article(body))
    assert sections[0] == Section(('First',), '')
    assert sections[1] == Section(('Second',), 'Body.')


def test_split_too_many_hashes_is_not_treated_as_heading():
    body = '####### Not A Heading\n\nBody.'
    sections = _split_by_markdown_heading(_make_article(body))
    assert len(sections) == 1
    assert sections[0].heading_path == ()
    assert '####### Not A Heading' in sections[0].text


def test_split_inline_hash_in_body_is_not_a_heading():
    body = '# Real Heading\n The #hashtag symbol is not a heading.'
    sections = _split_by_markdown_heading(_make_article(body))
    assert len(sections) == 1
    assert 'hashtag' in sections[0].text


def test_split_heading_title_has_no_trailing_carriage_return():
    body = '# Heading\r\n\r\nBody.'
    sections = _split_by_markdown_heading(_make_article(body))
    assert sections[0].title == 'Heading'


def test_split_whitespace_only_section_body_strips_to_empty():
    body = '# Heading\n\n   \n\n# Next\n\nBody.'
    sections = _split_by_markdown_heading(_make_article(body))
    assert sections[0].text == ''


def test_split_title_is_derived_from_heading_path():
    section = Section(('Parent', 'Child'), 'body')
    assert section.title == 'Child'
    assert Section((), 'body').title == ''


# ---------------------------------------------------------------
# goldens
# ---------------------------------------------------------------


@pytest.mark.parametrize('slug', GOLDEN_SLUGS)
def test_split_golden_preserves_all_body_text(slug):
    body, sections = _split_golden(slug)

    non_heading_body = _HEADING_RE.sub('', body)
    expected = ''.join(non_heading_body.split())
    actual = ''.join(''.join(s.text.split()) for s in sections)
    assert actual == expected


@pytest.mark.parametrize('slug', GOLDEN_SLUGS)
def test_split_golden_titles_match_heading_lines_in_order(slug):
    body, sections = _split_golden(slug)

    expected_titles = [m.group(2).strip() for m in _HEADING_RE.finditer(body)]
    actual_titles = [s.title for s in sections if s.heading_path]
    assert actual_titles == expected_titles


# Independent test does not share _HEADING_RE with the implementation
def test_split_golden_rod_of_wonder_matches_hand_written_sections():
    _, sections = _split_golden('magic-items__rods__rod-of-wonder')

    assert [s.heading_path for s in sections] == [
        ('Rod of Wonder',),
        ('Rod of Wonder', 'DESCRIPTION'),
        ('Rod of Wonder', 'CONSTRUCTION REQUIREMENTS'),
    ]
    assert sections[0].text.startswith('**Aura** moderate enchantment')
    assert sections[1].text.startswith('A rod of wonder is a strange')
    assert '| d% | Wondrous Effect |' in sections[1].text
    assert sections[2].text == (
        '**Feats** Craft Rod, confusion; **Special** creator must be chaotic; **Cost** 6,000 gp'
    )


_GOLDEN_SECTION_COUNTS = {
    'alignment-description__additional-rules': 90,
    'basics-ability-scores__glossary': 93,
    'bestiary__monster-listings__dragons__dragon__chromatic-red__adult-red-dragon': 9,
    'classes__core-classes__druid__animal-companions': 951,
    'classes__unchained-classes__rogue-unchained': 22,
    'equipment__weapons__weapon-descriptions__dagger': 1,
    'feats': 28,
    'feats__combat-feats__greater-two-weapon-fighting-combat': 1,
    'gamemastering__combat': 290,
    'magic-items__rods__rod-of-wonder': 3,
    'magic__all-spells__f__fireball': 5,
    'magic__spell-lists-and-domains__spell-lists-sorcerer-and-wizard': 4,
    'races__other-races__featured-races__arg-kobold': 19,
    'skills__stealth': 10,
    'traits__race-traits__firebug-kobold-red-scaled': 1,
}


@pytest.mark.parametrize('slug', GOLDEN_SLUGS)
def test_split_golden_section_count_is_stable(slug):
    _, sections = _split_golden(slug)
    assert len(sections) == _GOLDEN_SECTION_COUNTS[slug]


# ---------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------


def test_split_sentences_breaks_after_punctuation():
    assert _split_sentences('First one. Second one! Third one?') == ['First one.', 'Second one!', 'Third one?']


def test_split_sentences_does_not_break_on_missing_whitespace():
    assert _split_sentences('3.14 is pi.') == ['3.14 is pi.']


def test_split_sentences_drops_empty_and_strips():
    assert _split_sentences('  Only one.  ') == ['Only one.']
    assert _split_sentences('') == []


# ---------------------------------------------------------------
# _pack_lines
# ---------------------------------------------------------------


def test_pack_lines_greedily_fills_to_budget():
    lines = ['a a', 'b b', 'c c']  # two tokens each
    assert _pack_lines(lines, _tok, budget=4) == ['a a\nb b', 'c c']


def test_pack_lines_hard_splits_oversized_line_by_tokens():
    assert _pack_lines(['a b c d'], _tok, budget=2) == ['a b', 'c d']


def test_pack_lines_empty_input_returns_empty():
    assert _pack_lines([], _tok, budget=10) == []


# ---------------------------------------------------------------
# _pack_sentences
# ---------------------------------------------------------------


def test_pack_sentences_carries_overlap_into_next_window():
    sentences = ['Aa.', 'Bb.', 'Cc.', 'Dd.']
    assert _pack_sentences(sentences, _tok, budget=2, overlap=1) == ['Aa. Bb.', 'Bb. Cc.', 'Cc. Dd.']


def test_pack_sentences_no_overlap_does_not_repeat():
    sentences = ['Aa.', 'Bb.', 'Cc.', 'Dd.']
    assert _pack_sentences(sentences, _tok, budget=2, overlap=0) == ['Aa. Bb.', 'Cc. Dd.']


def test_pack_sentences_oversized_single_line_sentence_hard_splits():
    assert _pack_sentences(['one two three four'], _tok, budget=2, overlap=0) == ['one two', 'three four']


def test_pack_sentences_overlap_carry_respects_budget():
    # 9-token sentence leaves only 1 token of room: the 3-token carry candidate must be dropped
    sentences = ['a1 a2 a3.', 'b1 b2 b3.', 'c1 c2 c3 c4 c5 c6 c7 c8 c9.']
    windows = _pack_sentences(sentences, _tok, budget=10, overlap=4)
    assert windows == ['a1 a2 a3. b1 b2 b3.', 'c1 c2 c3 c4 c5 c6 c7 c8 c9.']


def test_pack_sentences_oversized_multiline_sentence_falls_back_to_lines():
    sentence = 'line aa\nline bb'
    assert _pack_sentences([sentence], _tok, budget=2, overlap=0) == ['line aa', 'line bb']


def test_pack_sentences_flushes_current_before_oversized_sentence():
    sentences = ['Short.', 'one two three four']
    assert _pack_sentences(sentences, _tok, budget=3, overlap=0) == ['Short.', 'one two three', 'four']


# ---------------------------------------------------------------
# _pack_table_rows
# ---------------------------------------------------------------


def test_pack_table_rows_repeats_header_on_each_split():
    block = '| a | b |\n| --- | --- |\n| r1 | r1 |\n| r2 | r2 |'
    splits = _pack_table_rows(block, _tok, budget=15)  # header is 10 tokens, each row 5. One row per split
    assert len(splits) == 2
    assert all(s.startswith('| a | b |\n| --- | --- |') for s in splits)
    assert '| r1 | r1 |' in splits[0]
    assert '| r2 | r2 |' in splits[1]


def test_pack_table_rows_header_only_block_returned_as_is():
    block = '| a | b |\n| --- | --- |'
    assert _pack_table_rows(block, _tok, budget=100) == [block]


def test_pack_table_rows_headerless_table_packs_rows_without_fake_header():
    # render_table produces no | --- | separator for tables without a <th> row
    # The first two data rows cannot be treated as a repeatable header
    block = '| r1 | r1 |\n| r2 | r2 |\n| r3 | r3 |\n| r4 | r4 |'
    splits = _pack_table_rows(block, _tok, budget=10)
    assert splits == ['| r1 | r1 |\n| r2 | r2 |', '| r3 | r3 |\n| r4 | r4 |']


def test_pack_table_rows_header_at_budget_falls_back_to_plain_line_packing():
    # header is 10 tokens -> whole budget. Repeating the header would double every split
    block = '| a | b |\n| --- | --- |\n| r1 | r1 |\n| r2 | r2 |'
    splits = _pack_table_rows(block, _tok, budget=10)
    assert splits == ['| a | b |\n| --- | --- |', '| r1 | r1 |\n| r2 | r2 |']


# ---------------------------------------------------------------
# _split_body
# ---------------------------------------------------------------


def test_split_body_routes_prose_and_table_and_preserves_order():
    text = 'Intro prose one.\n\n| h1 | h2 |\n| --- | --- |\n| a | b |\n\nOutro prose two.'
    bodies = _split_body(text, _tok, budget=50, overlap=0)
    assert bodies == [
        'Intro prose one.',
        '| h1 | h2 |\n| --- | --- |\n| a | b |',
        'Outro prose two.',
    ]


def test_split_body_processes_every_block_not_just_the_first():
    text = 'Block one.\n\nBlock two.\n\nBlock three.'
    bodies = _split_body(text, _tok, budget=100, overlap=0)
    assert bodies == ['Block one. Block two. Block three.']


def test_split_body_skips_blank_blocks():
    text = 'Only prose.\n\n\n\n   \n\n'
    assert _split_body(text, _tok, budget=100, overlap=0) == ['Only prose.']


# ---------------------------------------------------------------
# chunk_article
# ---------------------------------------------------------------


def test_chunk_article_one_chunk_per_small_section():
    article = _make_article('# Alpha\n\nOne two three.\n\n# Beta\n\nFour five.')
    chunks = chunk_article(article, _tok, max_tokens=50)

    assert [c.chunk_id for c in chunks] == ['bestiary__x__y#000', 'bestiary__x__y#001']
    assert chunks[0].text == 'Alpha\nOne two three.'
    assert chunks[1].text == 'Beta\nFour five.'
    assert chunks[0].heading_path == ['Alpha']
    assert chunks[0].n_tokens == 4


def test_chunk_article_splits_oversized_section_and_keeps_prefix():
    article = _make_article('# H\n\nAa bb. Cc dd. Ee ff.')
    chunks = chunk_article(article, _tok, max_tokens=6, overlap=0)

    assert len(chunks) == 2
    assert all(c.text.startswith('H\n') for c in chunks)
    assert [c.chunk_id for c in chunks] == ['bestiary__x__y#000', 'bestiary__x__y#001']


def test_chunk_article_ids_are_sequential_across_sections():
    article = _make_article('# H\n\nAa bb. Cc dd. Ee ff.\n\n# Tail\n\nDone.')
    chunks = chunk_article(article, _tok, max_tokens=6, overlap=0)

    assert [c.chunk_id for c in chunks] == [
        'bestiary__x__y#000',
        'bestiary__x__y#001',
        'bestiary__x__y#002',
    ]
    assert chunks[-1].text == 'Tail\nDone.'


def test_chunk_article_skips_empty_sections():
    article = _make_article('# Alpha\n\n## Beta\n\nBody here.')
    chunks = chunk_article(article, _tok, max_tokens=50)

    assert len(chunks) == 1
    assert chunks[0].text == 'Alpha > Beta\nBody here.'
    assert chunks[0].chunk_id == 'bestiary__x__y#000'


def test_chunk_article_empty_article_yields_no_chunks():
    assert chunk_article(_make_article(''), _tok, max_tokens=50) == []


def test_chunk_article_headingless_section_prefixed_with_article_title():
    article = _make_article('Preamble text here.\n\n# Alpha\n\nBody.')
    chunks = chunk_article(article, _tok, max_tokens=50)

    assert chunks[0].text == 'XY Title\nPreamble text here.'
    assert chunks[0].heading_path == []
    assert chunks[1].text == 'Alpha\nBody.'


def test_chunk_article_raises_when_heading_prefix_fills_max_tokens():
    article = _make_article('# one two three four five six seven eight\n\nBody text.')
    with pytest.raises(ValueError, match='heading prefix'):
        chunk_article(article, _tok, max_tokens=6)


# ---------------------------------------------------------------
# golden hard token limit + no text lost
# ---------------------------------------------------------------

_GOLDEN_MAX_TOKENS = 80
_GOLDEN_OVERLAP = 10
# BPE token counts are not additive across joins. The packer sums counts per piece but the embedder tokenizes as one
# string. The merges can push a window a token or two over budget which is deemed acceptable. see README design notes.
_TOKEN_DRIFT_SLACK = 2


@functools.cache
def _chunk_golden(slug: str):
    body, _ = _split_golden(slug)
    return chunk_article(_make_article(body), _tok, max_tokens=_GOLDEN_MAX_TOKENS, overlap=_GOLDEN_OVERLAP)


@pytest.mark.parametrize('slug', GOLDEN_SLUGS)
def test_chunk_golden_every_chunk_within_max_tokens(slug):
    chunks = _chunk_golden(slug)
    assert chunks
    assert all(c.n_tokens <= _GOLDEN_MAX_TOKENS + _TOKEN_DRIFT_SLACK for c in chunks)


@pytest.mark.parametrize('slug', GOLDEN_SLUGS)
def test_chunk_golden_chunks_reproduce_section_text_minus_overlaps(slug):
    """Section words must appear in order in the concatenated chunk bodies.
    Overlap carry and repeated table headers inserts duplicates -> order preserving subsequence check must detects
    any dropped text
    """
    _, sections = _split_golden(slug)
    chunks = _chunk_golden(slug)

    expected = [word for section in sections for word in section.text.split()]
    emitted = iter(word for chunk in chunks for word in chunk.text.split('\n', 1)[1].split())

    for word in expected:
        assert word in emitted, f'{slug}: lost text at {word!r}'
