import re
from functools import cache
from pathlib import Path

import lxml.html
import pytest

from rag.parsing import (
    _element_to_markdown,
    _normalize_whitespace,
    _retag_pseudo_headings,
    _slug_to_url,
    parse_page,
)

# TODO deferred to M1.4 (parse_corpus_dir), not this file's scope: hub-page filter (no '__' in slug),
# min_body_length filter, drop reasons logged


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def html_fragment(html: str) -> lxml.html.HtmlElement:
    return lxml.html.fromstring(html)


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
HTML_DIR = FIXTURES_DIR / 'HTML'
GOLDEN_DIR = FIXTURES_DIR / 'goldens'
FIXTURE_SLUGS = sorted(p.stem for p in HTML_DIR.glob('*.html'))


@cache
def _parsed_fixture(slug: str):
    html = (HTML_DIR / f'{slug}.html').read_text(encoding='utf-8')
    return parse_page(html, slug)


def _page(body: str, breadcrumbs: str = '<a href="/">Home</a>') -> str:
    return (
        '<html><body><div id="article-content">'
        f'<div class="breadcrumbs">{breadcrumbs}</div>'
        '<h1>Title</h1>'
        f'{body}'
        '</div></body></html>'
    )


# ---------------------------------------------------------------
# _normalize_whitespace
# ---------------------------------------------------------------


def test_normalize_whitespace_collapses_internal():
    assert _normalize_whitespace('a   b\n\n  c') == 'a b c'


def test_normalize_whitespace_none_returns_empty_string():
    assert _normalize_whitespace(None) == ''


# ---------------------------------------------------------------
# _element_to_markdown
# ---------------------------------------------------------------


# p/h paragraphs and headings
@pytest.mark.parametrize(
    ('html', 'expected'),
    [
        ('<div><p>Hello world</p></div>', 'Hello world'),
        ('<div><h1>Title</h1></div>', '# Title'),
        ('<div><h2>Title</h2></div>', '## Title'),
        ('<div><h3>Title</h3></div>', '### Title'),
        ('<div><h4>Title</h4></div>', '#### Title'),
        ('<div><h5>Title</h5></div>', '##### Title'),
        ('<div><h6>Title</h6></div>', '###### Title'),
    ],
)
def test_element_to_markdown_paragraph_and_headings(html, expected):
    assert _element_to_markdown(html_fragment(html)) == expected


# b/i formatting
@pytest.mark.parametrize(
    ('html', 'expected'),
    [
        ('<div><p>a <b>bold</b> b</p></div>', 'a **bold** b'),
        ('<div><p>a <strong>bold</strong> b</p></div>', 'a **bold** b'),
        ('<div><p>a <i>ital</i> b</p></div>', 'a *ital* b'),
        ('<div><p>a <em>ital</em> b</p></div>', 'a *ital* b'),
    ],
)
def test_element_to_markdown_bold_and_italic(html, expected):
    assert _element_to_markdown(html_fragment(html)) == expected


# li lists
def test_element_to_markdown_unordered_list():
    html = '<div><ul><li>One</li><li>Two</li></ul></div>'
    assert _element_to_markdown(html_fragment(html)) == '- One\n- Two'


def test_element_to_markdown_ordered_list():
    html = '<div><ol><li>One</li><li>Two</li></ol></div>'
    assert _element_to_markdown(html_fragment(html)) == '1. One\n2. Two'


def test_element_to_markdown_single_item_list():
    html = '<div><ol><li>Only</li></ol></div>'
    assert _element_to_markdown(html_fragment(html)) == '1. Only'


# tables
def test_element_to_markdown_table_with_header_separator():
    html = '<table><tr><th>H1</th><th>H2</th></tr><tr><td>a</td><td>b</td></tr></table>'
    assert _element_to_markdown(html_fragment(html)) == '| H1 | H2 |\n| --- | --- |\n| a | b |'


# br line breaks
def test_element_to_markdown_br_becomes_newline():
    html = '<div><p>Line1<br/>Line2</p></div>'
    assert _element_to_markdown(html_fragment(html)) == 'Line1\nLine2'


# block joining
def test_element_to_markdown_sibling_paragraphs_separated_by_blank_line():
    html = '<div><p>First</p><p>Second</p></div>'
    assert _element_to_markdown(html_fragment(html)) == 'First\n\nSecond'


def test_element_to_markdown_empty_paragraph_produces_empty_string():
    assert _element_to_markdown(html_fragment('<div><p></p></div>')) == ''


# fallback passthrough
def test_element_to_markdown_unknown_tag_passes_through_text():
    assert _element_to_markdown(html_fragment('<div><span>plain span</span></div>')) == 'plain span'


def test_element_to_markdown_anchor_keeps_text_only_no_link_markup():
    assert _element_to_markdown(html_fragment('<div><a href="x">link text</a></div>')) == 'link text'


# whitespace
def test_element_to_markdown_normalizes_internal_whitespace():
    html = '<div><p>  extra   spaces\n  here </p></div>'
    assert _element_to_markdown(html_fragment(html)) == 'extra spaces here'


# skipped comments keep tail
def test_element_to_markdown_comment_is_skipped_but_tail_kept():
    html = '<div>Start <!-- hidden --> End</div>'
    result = _element_to_markdown(html_fragment(html))
    assert 'hidden' not in result
    assert 'Start' in result
    assert 'End' in result


# nested lists / tables. bug fixes


def test_element_to_markdown_ordered_list_separates_from_prior_text():
    html = '<div><p>Intro</p><ol><li>One</li></ol></div>'
    assert _element_to_markdown(html_fragment(html)) == 'Intro\n\n1. One'


def test_element_to_markdown_bold_survives_in_list_item():
    html = '<div><ul><li>Deal <b>2d6</b> fire</li></ul></div>'
    assert _element_to_markdown(html_fragment(html)) == '- Deal **2d6** fire'


def test_element_to_markdown_br_survives_in_list_item():
    html = '<div><ul><li>a<br/>b</li></ul></div>'
    assert _element_to_markdown(html_fragment(html)) == '- a\nb'


def test_element_to_markdown_nested_list_indents_one_level():
    html = '<div><ul><li>Parent<ul><li>Child</li></ul></li></ul></div>'
    assert _element_to_markdown(html_fragment(html)) == '- Parent\n  - Child'


def test_element_to_markdown_nested_ordered_list_indents_to_marker_width():
    html = '<div><ol><li>Parent<ol><li>Child</li></ol></li></ol></div>'
    assert _element_to_markdown(html_fragment(html)) == '1. Parent\n   1. Child'


def test_element_to_markdown_li_starting_with_nested_list_no_double_marker():
    html = '<div><ul><li><ul><li>a</li></ul></li></ul></div>'
    assert _element_to_markdown(html_fragment(html)) == '- a'


def test_element_to_markdown_comment_in_list_is_skipped_not_bulleted():
    html = '<div><ul><!-- nav --><li>real</li></ul></div>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '- real'


def test_element_to_markdown_comment_in_ordered_list_keeps_numbering_honest():
    html = '<div><ol><!-- nav --><li>One</li><li>Two</li></ol></div>'
    assert _element_to_markdown(html_fragment(html)) == '1. One\n2. Two'


def test_element_to_markdown_table_without_th_gets_no_header_separator():
    html = '<table><tr><td>5</td><td>10</td></tr><tr><td>15</td><td>20</td></tr></table>'
    assert _element_to_markdown(html_fragment(html)) == '| 5 | 10 |\n| 15 | 20 |'


def test_element_to_markdown_table_with_tbody_renders():
    html = '<table><tbody><tr><th>H1</th><th>H2</th></tr><tr><td>a</td><td>b</td></tr></tbody></table>'
    assert _element_to_markdown(html_fragment(html)) == '| H1 | H2 |\n| --- | --- |\n| a | b |'


def test_element_to_markdown_layout_table_renders_cells_as_blocks():
    html = '<table><tr><td>outer<table><tr><td>label</td><td>inner</td></tr></table></td></tr></table>'
    result = _element_to_markdown(html_fragment(html))
    assert result == 'outer\n\n| label | inner |'


def test_element_to_markdown_layout_table_keeps_cell_leading_text():
    html = (
        '<table><tr><td>Notes: '
        '<table><tr><th>K</th><th>V</th></tr><tr><td>Racial</td><td>+6</td></tr></table>'
        '</td></tr></table>'
    )
    result = _element_to_markdown(html_fragment(html))
    assert result == 'Notes:\n\n| K | V |\n| --- | --- |\n| Racial | +6 |'


def test_element_to_markdown_bold_survives_in_table_cell():
    html = '<table><tr><td>DC <b>15</b></td></tr></table>'
    assert _element_to_markdown(html_fragment(html)) == '| DC **15** |'


def test_element_to_markdown_br_in_table_cell_does_not_break_row():
    html = '<table><tr><td>a<br/>b</td></tr></table>'
    assert _element_to_markdown(html_fragment(html)) == '| a b |'


def test_element_to_markdown_div_after_table_stays_separated():
    html = '<div><table><tr><td>a</td></tr></table><div>caption</div></div>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '| a |\n\ncaption'


# styled-p pseudo-headings
def test_element_to_markdown_bold_border_bottom_p_becomes_h5():
    html = '<div><p style="font-weight:bold;border-bottom:solid thin">Opportunities and Allies</p></div>'
    assert _element_to_markdown(html_fragment(html)) == '##### Opportunities and Allies'


def test_element_to_markdown_bold_border_bottom_style_spelling_variant_becomes_h5():
    html = '<div><p style="font-weight: bold;border-bottom-style: solid">Variant</p></div>'
    assert _element_to_markdown(html_fragment(html)) == '##### Variant'


def test_element_to_markdown_plain_bold_p_without_border_stays_text():
    html = '<div><p style="font-weight:bold">Just bold</p></div>'
    assert _element_to_markdown(html_fragment(html)) == 'Just bold'


def test_element_to_markdown_p_title_becomes_h2():
    html = '<div><p class="title">Aboleth</p></div>'
    assert _element_to_markdown(html_fragment(html)) == '## Aboleth'


# tables: caption, colspan, malformed markup
def test_element_to_markdown_table_caption_becomes_bold_line_above_table():
    html = '<table><caption>My Caption</caption><tr><th>H1</th></tr><tr><td>a</td></tr></table>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '**My Caption**\n\n| H1 |\n| --- |\n| a |'


def test_element_to_markdown_colspan_pads_cells_to_agree_with_row_width():
    html = '<table><tr><th colspan="3">Lawful</th></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '| Lawful |  |  |\n| --- | --- | --- |\n| a | b | c |'


def test_element_to_markdown_garbage_colspan_falls_back_to_one():
    html = '<table><tr><th colspan="garbage">H</th></tr><tr><td>a</td></tr></table>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '| H |\n| --- |\n| a |'


def test_element_to_markdown_malformed_table_unclosed_tbody_keeps_all_rows():
    # source HTML has an unclosed <tbody>; lxml's error recovery nests a second
    # <thead> inside the first <tbody> instead of dropping the row
    html = (
        '<table><thead><tr><th>H1</th><th>H2</th></tr></thead>'
        '<tbody><thead><tr><td>a</td><td>b</td></tr></thead>'
        '<tr><td>c</td><td>d</td></tr></tbody></table>'
    )
    result = _element_to_markdown(html_fragment(html))
    assert result == '| H1 | H2 |\n| --- | --- |\n| a | b |\n| c | d |'


def test_element_to_markdown_literal_less_than_escaped_not_parsed_as_tag():
    html = '<div><p>Text with &lt;a protean&gt; creature</p></div>'
    result = _element_to_markdown(html_fragment(html))
    assert result == 'Text with \\<a protean> creature'


# ---------------------------------------------------------------
# parse_page
# ---------------------------------------------------------------


def test_parse_page_extracts_breadcrumbs_from_anchors():
    html = _page('<p>Body</p>', breadcrumbs='<a href="/">Home</a><a href="/x">X</a>')
    article = parse_page(html, 'x')
    assert article.breadcrumb == ['Home', 'X']


def test_parse_page_title_from_h1_text_content_with_nested_markup_and_whitespace():
    html = (
        '<html><body><div id="article-content">'
        '<div class="breadcrumbs"><a href="/">Home</a></div>'
        '<h1>  Red <b>Dragon</b>, Adult  </h1>'
        '<p>Body</p>'
        '</div></body></html>'
    )
    article = parse_page(html, 'x')
    assert article.title == 'Red Dragon, Adult'


JUNK_CLASSES = (
    'section15',
    'toc_light_blue',
    'goog-toc',
    'product-right',
    'ogn-childpages',
    'custom-content',
    'sites-comment-docos-wrapper',
    'right-sidebar',
    'sidebar-bottom',
    'footer-nav',
    'container-fluid',
    'ogn-npa-container',
)


@pytest.mark.parametrize('junk_class', JUNK_CLASSES)
def test_parse_page_strips_junk_class(junk_class):
    html = _page(f'<p>Keep me</p><div class="{junk_class}">JUNK_MARKER</div>')
    article = parse_page(html, 'x')
    assert 'JUNK_MARKER' not in article.body_md
    assert 'Keep me' in article.body_md


def test_parse_page_removes_script_tags():
    html = _page('<p>Keep me</p><script>var evil = "JUNK_MARKER";</script>')
    article = parse_page(html, 'x')
    assert 'JUNK_MARKER' not in article.body_md


def test_parse_page_raises_value_error_with_slug_for_malformed_html():
    with pytest.raises(ValueError, match='my-slug'):
        parse_page('', 'my-slug')


# ---------------------------------------------------------------
# _slug_to_url
# ---------------------------------------------------------------


def test_slug_to_url_round_trip_nested_segments():
    assert str(_slug_to_url('bestiary__x__y')) == 'https://www.d20pfsrd.com/bestiary/x/y'


def test_slug_to_url_strips_html_suffix():
    assert str(_slug_to_url('bestiary__x.html')) == 'https://www.d20pfsrd.com/bestiary/x'


def test_slug_to_url_index_becomes_base_url_with_trailing_slash():
    assert str(_slug_to_url('index')) == 'https://www.d20pfsrd.com/'


def test_slug_to_url_hashed_slug_raises_value_error():
    with pytest.raises(ValueError, match='truncated and hashed'):
        _slug_to_url('bestiary__monster-listings__c35fd7cde8')


def test_slug_to_url_single_underscore_segment_survives():
    assert str(_slug_to_url('a_b__c')) == 'https://www.d20pfsrd.com/a_b/c'


# ---------------------------------------------------------------
# _retag_pseudo_headings
# ---------------------------------------------------------------


def test_retag_pseudo_headings_divider_under_title_becomes_h3():
    html = '<div><p class="title">Aboleth</p><p class="divider">DEFENSE</p></div>'
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    assert [c.tag for c in root] == ['p', 'h3']


def test_retag_pseudo_headings_divider_under_h4_becomes_h5():
    html = '<div><h4>Starting Statistics</h4><p class="divider">FOO</p></div>'
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    assert [c.tag for c in root] == ['h4', 'h5']


def test_retag_pseudo_headings_divider_keeps_rank_past_deeper_headings():
    html = (
        '<div>'
        '<p class="title">Monster</p>'
        '<p class="divider">SPECIAL ABILITIES</p>'
        '<h4>Some Ability</h4>'
        '<p class="divider">ECOLOGY</p>'
        '</div>'
    )
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    assert [c.tag for c in root] == ['p', 'h3', 'h4', 'h3']


def test_retag_pseudo_headings_divider_level_caps_at_h6():
    html = '<div><h6>Deep</h6><p class="divider">FOO</p></div>'
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    assert [c.tag for c in root] == ['h6', 'h6']


def test_retag_pseudo_headings_ed_note_header_becomes_h5():
    html = '<div><div class="ed-note-header">Note text</div></div>'
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    assert root[0].tag == 'h5'


@pytest.mark.parametrize('sidebar_class', ('content-sidebar', 'info-sidebar', 'faq', 'widefaq'))
@pytest.mark.parametrize('title_tag', ('div', 'p'))
def test_retag_pseudo_headings_box_title_becomes_h5(sidebar_class, title_tag):
    html = f'<div><div class="{sidebar_class}"><{title_tag}>Box Title</{title_tag}><p>body</p></div></div>'
    root = html_fragment(html)
    _retag_pseudo_headings(root)
    box = root[0]
    assert box[0].tag == 'h5'
    assert box[0].text_content() == 'Box Title'


# ---------------------------------------------------------------
# golden tests: parse_page(html, slug).body_md vs goldens/<slug>.golden.md hand audited vs fixtures
# ---------------------------------------------------------------


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_parse_page_body_matches_golden(slug):
    html = (HTML_DIR / f'{slug}.html').read_text(encoding='utf-8')
    golden = (GOLDEN_DIR / f'{slug}.golden.md').read_text(encoding='utf-8')
    assert parse_page(html, slug).body_md == golden


# ---------------------------------------------------------------
# invariants. parametrized over all fixtures
# ---------------------------------------------------------------

_UNESCAPED_TAG_RE = re.compile(r'(?<!\\)</?[a-z]')


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_golden_body_has_no_unescaped_html_tags(slug):
    # "'<' not in body" is wrong. Source text  contains <> ex:  '<Oozes are…>'
    assert _UNESCAPED_TAG_RE.search(_parsed_fixture(slug).body_md) is None


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_golden_body_has_no_ogl(slug):
    assert 'OPEN GAME LICENSE' not in _parsed_fixture(slug).body_md


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_golden_body_has_no_subpages_nav(slug):
    assert 'Subpages' not in _parsed_fixture(slug).body_md


def _pipe_blocks(body_md: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in body_md.split('\n'):
        if line.startswith('|'):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _is_separator_row(row: str) -> bool:
    return set(row) <= set('|- ')


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_golden_table_rows_match_header_cell_count(slug):
    body = _parsed_fixture(slug).body_md
    for block in _pipe_blocks(body):
        header_cell_count = block[0].count('|') - 1
        for row in block:
            if _is_separator_row(row):
                continue
            assert row.count('|') - 1 == header_cell_count


@pytest.mark.parametrize('slug', FIXTURE_SLUGS)
def test_golden_article_metadata_invariants(slug):
    article = _parsed_fixture(slug)
    assert article.title != ''
    assert article.category == slug.split('__')[0]
    assert article.n_chars == len(article.body_md)
