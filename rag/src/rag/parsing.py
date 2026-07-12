import re
from collections.abc import Callable
from dataclasses import dataclass

import lxml.html

from rag.models import Article

# ---------------------------------------------------------------
# HTML parsing tag registry
# ---------------------------------------------------------------


@dataclass
class TagSpec:
    """HTML tags"""

    is_block: bool
    render: Callable[[lxml.html.HtmlElement, str], str]


TAGS: dict[str, TagSpec] = {}


def register(name: str, is_block: bool):
    def wrapper(fn: Callable[[lxml.html.HtmlElement, str], str]):
        TAGS[name] = TagSpec(is_block=is_block, render=fn)
        return fn

    return wrapper


# ---------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------

_WHITESPACE_RE = re.compile(r'\s+')


def _normalize_whitespace(text: str | None) -> str:
    if text is None:
        return ''
    return _WHITESPACE_RE.sub(' ', text)


def _render_li(li: lxml.html.HtmlElement, marker: str) -> str:
    """renders on <li> tag, indenting one level of nested if ul/ol present"""
    indent = ' ' * len(marker)
    text_parts = [_normalize_whitespace(li.text)]

    for child in li:
        child_markdown = _element_to_markdown(child)

        if child.tag in ('ul', 'ol'):
            indented = '\n'.join(indent + line for line in child_markdown.strip().split('\n'))
            text_parts.append('\n' + indented)
        else:
            text_parts.append(child_markdown)
    return ''.join(text_parts).strip()


def _direct_rows(table: lxml.html.HtmlElement):
    """Yield the table's own <tr> rows. Not rows from any table nested inside a cell."""
    for child in table:
        if child.tag == 'tr':
            yield child
        elif child.tag in ('thead', 'tbody', 'tfoot'):
            for tr in child:
                if tr.tag == 'tr':
                    yield tr


def _flatten_nested_table(table: lxml.html.HtmlElement) -> str:
    """Markdown cannot do nested tables. Render nested table as inline 'label: value; label: value' text."""
    pairs = []
    for tr in _direct_rows(table):
        cells = [_normalize_whitespace(cell.text_content()).strip() for cell in tr if cell.tag in ('th', 'td')]
        if len(cells) >= 2:
            pairs.append(f'{cells[0]}: {cells[1]}')
    return '; '.join(pairs)


def _wrap(element: lxml.html.HtmlElement, content: str) -> str:
    # wrapped known tags in markdown equivalent
    tag = element.tag

    wrapped_content = content.strip()

    if tag in TAGS:
        return TAGS[tag].render(element, wrapped_content)
    else:
        return wrapped_content


def _element_to_markdown(element: lxml.html.HtmlElement) -> str:
    if isinstance(element, lxml.html.HtmlComment):
        return _normalize_whitespace(element.tail)

    parts = [_normalize_whitespace(element.text)]
    has_content = bool(parts[0].strip())

    for child in element:
        child_markdown = _element_to_markdown(child)

        spec = TAGS.get(child.tag)
        if spec is not None and spec.is_block and has_content:
            parts[-1] = parts[-1].rstrip()
            parts.append('\n\n' + child_markdown.lstrip())
        else:
            parts.append(child_markdown)

        if child_markdown.strip():
            has_content = True

    assembled = _wrap(element, ''.join(parts))
    return assembled + _normalize_whitespace(element.tail)


# ---------------------------------------------------------------
# parse_corpus
# ---------------------------------------------------------------


def parse_corpus(raw_text: str, min_body_length: int = 40) -> list[Article]:
    """Parse the scraped markdown corpus into article dicts. TODO: implement."""
    raise NotImplementedError('parse_corpus not yet implemented')


# ---------------------------------------------------------------
# tag renderers
# ---------------------------------------------------------------


@register('div', is_block=True)
def render_div(element: lxml.html.HtmlElement, content: str) -> str:
    return content


@register('p', is_block=True)
def render_p(element: lxml.html.HtmlElement, content: str) -> str:
    return content


@register('h1', is_block=True)
def render_h1(element: lxml.html.HtmlElement, content: str) -> str:
    return '# ' + content


@register('h2', is_block=True)
def render_h2(element: lxml.html.HtmlElement, content: str) -> str:
    return '## ' + content


@register('h3', is_block=True)
def render_h3(element: lxml.html.HtmlElement, content: str) -> str:
    return '### ' + content


@register('h4', is_block=True)
def render_h4(element: lxml.html.HtmlElement, content: str) -> str:
    return '#### ' + content


@register('h5', is_block=True)
def render_h5(element: lxml.html.HtmlElement, content: str) -> str:
    return '##### ' + content


@register('h6', is_block=True)
def render_h6(element: lxml.html.HtmlElement, content: str) -> str:
    return '###### ' + content


@register('b', is_block=False)
def render_b(element: lxml.html.HtmlElement, content: str) -> str:
    return '**' + content + '**'


@register('strong', is_block=False)
def render_strong(element: lxml.html.HtmlElement, content: str) -> str:
    return '**' + content + '**'


@register('i', is_block=False)
def render_i(element: lxml.html.HtmlElement, content: str) -> str:
    return '*' + content + '*'


@register('em', is_block=False)
def render_em(element: lxml.html.HtmlElement, content: str) -> str:
    return '*' + content + '*'


@register('br', is_block=False)
def render_br(element: lxml.html.HtmlElement, content: str) -> str:
    return '\n'


@register('ul', is_block=True)
def render_ul(element: lxml.html.HtmlElement, content: str) -> str:
    lines = []
    for li in element.findall('li'):
        marker = '- '
        rendered = _render_li(li, marker)
        if rendered.startswith(marker):
            lines.append(rendered)
        else:
            lines.append(marker + rendered)
    return '\n'.join(lines)


@register('ol', is_block=True)
def render_ol(element: lxml.html.HtmlElement, content: str) -> str:
    lines = []
    for i, li in enumerate(element.findall('li'), start=1):
        marker = f'{i}. '
        rendered = _render_li(li, marker)
        if rendered.startswith(marker):
            lines.append(rendered)
        else:
            lines.append(marker + rendered)
    return '\n'.join(lines)


@register('table', is_block=True)
def render_table(element: lxml.html.HtmlElement, content: str) -> str:
    rows = []
    for tr in _direct_rows(element):
        cells = []
        for cell in tr:
            if cell.tag in ('th', 'td'):
                nested_table = cell.find('table')
                if nested_table is not None:
                    own_text = _normalize_whitespace(cell.text).strip()
                    flattened = _flatten_nested_table(nested_table)
                    cells.append((own_text + ' ' + flattened).strip() if own_text else flattened)
                else:
                    cells.append(' '.join(_element_to_markdown(cell).split()))
        rows.append('| ' + ' | '.join(cells) + ' |')

        if len(rows) == 1 and any(child.tag == 'th' for child in tr):
            sep = ['---'] * len(cells)
            rows.append('| ' + ' | '.join(sep) + ' |')

    return '\n'.join(rows)
