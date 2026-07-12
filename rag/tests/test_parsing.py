import lxml.html
import pytest

from rag.parsing import _element_to_markdown, _normalize_whitespace

# TODO make test corpus fixture
# TODO test skip empty bodies
# TODO test removing junk patterns ?
# TODO make a list off parsings

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def html_fragment(html: str) -> lxml.html.HtmlElement:
    return lxml.html.fromstring(html)


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


def test_element_to_markdown_div_after_table_stays_separated():
    html = '<div><table><tr><td>a</td></tr></table><div>caption</div></div>'
    result = _element_to_markdown(html_fragment(html))
    assert result == '| a |\n\ncaption'
