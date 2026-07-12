import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import lxml.html
from pydantic import HttpUrl

from rag.models import Article

logger = logging.getLogger(__name__)

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

        spec = TAGS.get(child.tag) if isinstance(child.tag, str) else None
        if spec is not None and spec.is_block and has_content:
            parts[-1] = parts[-1].rstrip()
            parts.append('\n\n' + child_markdown.lstrip())
        else:
            parts.append(child_markdown)

        if child_markdown.strip():
            has_content = True

    assembled = _wrap(element, ''.join(parts))
    return assembled + _normalize_whitespace(element.tail)


BASE_URL = 'https://www.d20pfsrd.com'
_HASH_SUFFIX_RE = re.compile(r'__[0-9a-f]{10}$')


def _slug_to_url(slug: str) -> HttpUrl:
    if slug.endswith('.html'):
        slug = slug[: -len('.html')]

    if slug == 'index':
        return HttpUrl(BASE_URL + '/')

    if _HASH_SUFFIX_RE.search(slug):
        raise ValueError(f'Slug {slug!r} was truncated and hashed by url_to_filename')

    path = slug.replace('__', '/')
    return HttpUrl(f'{BASE_URL}/{path}')


# ---------------------------------------------------------------
# parse_page
# ---------------------------------------------------------------


def parse_page(html: str, slug: str) -> Article:
    try:
        root = lxml.html.fromstring(html)
    except Exception as e:
        raise ValueError(f'Failed to parse HTML for slug {slug!r}: {e}') from e

    content = root.get_element_by_id('article-content')

    # Save page breadcrumbs for metadata as list[str]
    breadcrumbs_div = content.find_class('breadcrumbs')[0]
    breadcrumb = [a.text_content().strip() for a in breadcrumbs_div.iter('a')]

    # Remove unneeded in tree: breadcrumbs, OGL section 15, TOC, Shopify/product etc. drop_tree keeps .tail
    for classes in ('breadcrumbs', 'section15', 'toc_light_blue', 'product-right'):
        for element in content.find_class(classes):
            element.drop_tree()
    # remove scripts
    for element in list(content.iter('script')):
        if element.getparent() is not None:
            element.drop_tree()

    title = ''
    for child in content:
        if child.tag == 'h1':
            title = child.text_content().strip()
            break

    category = slug.split('__')[0]

    body = _element_to_markdown(content)
    n_chars = len(body)

    return Article(
        doc_id=slug,
        url=_slug_to_url(slug),
        title=title,
        category=category,
        breadcrumb=breadcrumb,
        body_md=body,
        n_chars=n_chars,
    )


# ---------------------------------------------------------------
# tag renderers
# ---------------------------------------------------------------


@register('div', is_block=True)
def render_div(element: lxml.html.HtmlElement, content: str) -> str:
    return content


# special cases f or p.title,  p.divider in bestiary
@register('p', is_block=True)
def render_p(element: lxml.html.HtmlElement, content: str) -> str:
    classes = element.get('class', '').split()
    if 'title' in classes:
        return '## ' + content
    if 'divider' in classes:
        return '### ' + content
    # catch CSS third heading in CSS <p style="font-weight: bold;border-bottom: solid thin">Opportunities and Allies</p
    style = _WHITESPACE_RE.sub('', element.get('style', '').lower())
    if 'font-weight:bold' in style and 'border-bottom' in style:
        return '##### ' + content
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


def _colspan(cell: lxml.html.HtmlElement) -> int:
    try:
        return max(1, int(cell.get('colspan') or 1))
    except ValueError:
        return 1


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
                # pad so header and body rows agree on column count
                cells.extend([''] * (_colspan(cell) - 1))
        rows.append('| ' + ' | '.join(cells) + ' |')

        if len(rows) == 1 and any(child.tag == 'th' for child in tr):
            sep = ['---'] * len(cells)
            rows.append('| ' + ' | '.join(sep) + ' |')

    table_md = '\n'.join(rows)

    caption = element.find('caption')
    if caption is not None:
        caption_text = _normalize_whitespace(caption.text_content()).strip()
        if caption_text:
            return '**' + caption_text + '**\n\n' + table_md
    return table_md
