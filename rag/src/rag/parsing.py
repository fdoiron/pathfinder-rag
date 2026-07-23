import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import lxml.html
from pydantic import HttpUrl

from rag.models import Article

logger = logging.getLogger(__name__)

# ChunksManifest record of changes to parsing
PARSER_VERSION = '1'

# ---------------------------------------------------------------
# HTML parsing tag registry
# ---------------------------------------------------------------


@dataclass
class TagSpec:
    """HTML tags"""

    is_block: bool
    render: Callable[[lxml.html.HtmlElement, str], str]


TAGS: dict[str, TagSpec] = {}


RenderFn = Callable[[lxml.html.HtmlElement, str], str]


def register(name: str, is_block: bool) -> Callable[[RenderFn], RenderFn]:
    def wrapper(fn: RenderFn) -> RenderFn:
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
    # sanitize literal '<' int text
    # ex: '<a protean>' mistaken for an HTML tag
    return _WHITESPACE_RE.sub(' ', text).replace('<', '\\<')


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


def _direct_rows(table: lxml.html.HtmlElement) -> Iterator[lxml.html.HtmlElement]:
    """Yield table's own <tr> rows.
    Not rows from any table nested inside a cell

    Bugfix on HTML malformed in source:
    Recurses thead/tbody/tfoot because malformed source HTML (unclosed tbody)
    makes lxml nested row groups inside each other during error recovery
    """
    for child in table:
        if child.tag == 'tr':
            yield child
        elif child.tag in ('thead', 'tbody', 'tfoot'):
            yield from _direct_rows(child)


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

    tail = _normalize_whitespace(element.tail)
    spec = TAGS.get(element.tag) if isinstance(element.tag, str) else None
    if spec is not None and spec.is_block and tail.strip():
        # Separate block element from trailing text so heading's tail does not merge onto the heading line
        # (## TitleText -> ## Title\n\nText).
        return assembled.rstrip() + '\n\n' + tail.lstrip()
    return assembled + tail


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

_HEADING_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')


def _retag_pseudo_headings(content: lxml.html.HtmlElement) -> None:
    """transforms heading markup into h* tags before conversion

    - info-box titles (editor notes, content-sidebar) -> h5
    - p.divider  -> one level below the nearest preceding heading because the same divider sits under
    p.title (##) in bestiary but under h4 in class pages
    """
    for note_header in content.find_class('ed-note-header'):
        note_header.tag = 'h5'
    for sidebar_class in ('content-sidebar', 'info-sidebar', 'faq', 'widefaq'):
        for sidebar in content.find_class(sidebar_class):
            first = next(iter(sidebar), None)
            # box title usually a bare <div> but occasionally is <p> depending on page author
            if first is not None and first.tag in ('div', 'p'):
                first.tag = 'h5'

    level = 1
    divider_level: int | None = None
    for el in content.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag in _HEADING_TAGS:
            heading_level = int(el.tag[1])
            # headings deeper than the divider rank are under a divider
            # (ex: h4 ability names inside SPECIAL ABILITIES)
            if divider_level is None or heading_level < divider_level:
                level = heading_level
                divider_level = None
        elif el.tag == 'p':
            classes = (el.get('class') or '').split()
            if 'title' in classes:
                if divider_level is None or divider_level > 2:
                    level = 2
                    divider_level = None
            elif 'divider' in classes:
                if divider_level is None:
                    divider_level = min(level + 1, 6)
                el.tag = f'h{divider_level}'


def parse_page(html: str, slug: str) -> Article:
    try:
        root = lxml.html.fromstring(html)
        content = root.get_element_by_id('article-content')
        # Save page breadcrumbs for metadata as list[str]
        breadcrumbs_div = content.find_class('breadcrumbs')[0]
    except Exception as e:
        # covers malformed HTML plus missing #article-content or div.breadcrumbs
        # (page doesn't match the expected site template)
        raise ValueError(f'Failed to parse HTML for slug {slug!r}: {e}') from e

    breadcrumb = [a.text_content().strip() for a in breadcrumbs_div.iter('a')]

    # Remove unneeded in tree, drop_tree keeps .tail.
    # breadcrumbs (metadata, saved), section15 (OGL), toc_light_blue/goog-toc (TOCs),
    # product-right (Shopify), ogn-childpages (Subpages nav), custom-content (,out of scope),
    # sites-comment-docos-wrapper (Google Sites comments) and malformed pages
    junk_classes = (
        'breadcrumbs',
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
    for classes in junk_classes:
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

    _retag_pseudo_headings(content)

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
# corpus directory parser
# ---------------------------------------------------------------


def parse_corpus_dir(html_dir: Path, min_body_length: int) -> list[Article]:
    articles: list[Article] = []
    for html_file in sorted(html_dir.glob('*.html')):
        slug = html_file.stem

        try:
            article = parse_page(html_file.read_text(encoding='utf-8'), slug)
        except ValueError as e:
            logger.warning(f'dropped {slug!r}: parse error: {e}')
            continue

        if article.n_chars < min_body_length:
            logger.warning(f'dropped {slug!r}: body too short ({article.n_chars} < {min_body_length} chars)')
            continue

        articles.append(article)
    return articles


# ---------------------------------------------------------------
# tag renderers
# ---------------------------------------------------------------


@register('div', is_block=True)
def render_div(element: lxml.html.HtmlElement, content: str) -> str:
    return content


# special cases for p.title, p.divider in bestiary
@register('p', is_block=True)
def render_p(element: lxml.html.HtmlElement, content: str) -> str:
    classes = element.get('class', '').split()
    if 'title' in classes:
        return '## ' + content
    if 'divider' in classes:
        return '### ' + content
    # catches CSS third heading in CSS <p style="font-weight: bold;border-bottom: solid thin">Opportunities and Allies</
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


def _span(cell: lxml.html.HtmlElement, attr: str) -> int:
    try:
        return max(1, int(cell.get(attr) or 1))
    except ValueError:
        return 1


# layout containers, not data: cells render as flowing blocks (see render_table)
@register('tr', is_block=True)
def render_tr(element: lxml.html.HtmlElement, content: str) -> str:
    return content


@register('td', is_block=True)
def render_td(element: lxml.html.HtmlElement, content: str) -> str:
    return content


@register('th', is_block=True)
def render_th(element: lxml.html.HtmlElement, content: str) -> str:
    return content


@register('table', is_block=True)
def render_table(element: lxml.html.HtmlElement, content: str) -> str:
    # table containing another table is layout wrapper (site-wide, nested tables only happen as borderless containers
    # around real data tables): render cells as normal block flow instead of a pipe table
    if element.find('.//table') is not None:
        return content

    # pad colspan and rowspan with empty cells so every row is the same as the header's column count.
    # pending maps column index -> rows still covered by a rowspan cell above
    cell_rows: list[list[str]] = []
    first_row_has_th = False
    pending: dict[int, int] = {}
    for tr in _direct_rows(element):
        cells: list[str] = []
        for cell in tr:
            if cell.tag in ('th', 'td'):
                while pending.get(len(cells), 0) > 0:
                    pending[len(cells)] -= 1
                    cells.append('')
                rendered = ' '.join(_element_to_markdown(cell).split())
                rowspan = _span(cell, 'rowspan')
                for i in range(_span(cell, 'colspan')):
                    if rowspan > 1:
                        pending[len(cells)] = rowspan - 1
                    cells.append(rendered if i == 0 else '')
        while pending.get(len(cells), 0) > 0:
            pending[len(cells)] -= 1
            cells.append('')
        if not cell_rows:
            first_row_has_th = any(child.tag == 'th' for child in tr)
        cell_rows.append(cells)

    # page miscount colspans : rogue-unchained golden declares 39 over 38 leafcolumns
    # Ragged rows happen even after span padding. Normalize every row to the widest row
    width = max((len(cells) for cells in cell_rows), default=0)
    rows = []
    # known limitation: a literal '|' in cell text is not escaped, so it would split into extra pipe-table columns.
    # Not observed in the corpus so far.
    # known limitation: a table with no <th> row gets no '---' separator, so GFM viewers (ex: GitHub) render it
    # as plain text instead of a table. Harmless for embedding/retrieval.
    for cells in cell_rows:
        padded = cells + [''] * (width - len(cells))
        rows.append('| ' + ' | '.join(padded) + ' |')
        if len(rows) == 1 and first_row_has_th:
            rows.append('| ' + ' | '.join(['---'] * width) + ' |')

    table_md = '\n'.join(rows)

    caption = element.find('caption')
    if caption is not None:
        caption_text = _normalize_whitespace(caption.text_content()).strip()
        if caption_text:
            return '**' + caption_text + '**\n\n' + table_md
    return table_md
