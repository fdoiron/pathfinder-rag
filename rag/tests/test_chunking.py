import functools
from pathlib import Path

import pytest

from rag.chunking import _HEADING_RE, Section, _split_by_markdown_heading
from rag.models import Article

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


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
