import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Any, Protocol, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rag.models import Article, Chunk

_HEADING_RE = re.compile(r'^(#{1,6})[ \t]+(.*)$', re.MULTILINE)

# The packer counts tokens by summing each piece's token counts to decide if a window is full.
# However, the tokenizer does it across one string, which can cause the total to be more than the sum of its parts.
# Reserve that headroom in the body budget up front so the assembled chunk stays within max_tokens

_DRIFT_MARGIN = 0.02


class Tokenizer(Protocol):
    """The subset of PreTrainedTokenizerBase the packer needs so tests can use a lightweight fake"""

    def __call__(self, text: str, add_special_tokens: bool = ...) -> Mapping[str, Sequence[Any]]: ...
    def decode(self, ids: Any, /) -> str | list[str]: ...


@lru_cache
def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(name))


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    text: str

    @property
    def title(self) -> str:
        return self.heading_path[-1] if self.heading_path else ''


def _split_by_markdown_heading(page: Article) -> list[Section]:
    body = page.body_md

    matches = list(_HEADING_RE.finditer(body))

    # No headings in article
    if not matches:
        return [Section((), body.strip())]

    sections: list[Section] = []

    # Article starts with a no heading preamble section
    if matches[0].start() != 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(Section((), preamble))

    stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        heading_level = len(match.group(1))
        title = match.group(2).strip()

        while stack and stack[-1][0] >= heading_level:
            stack.pop()
        stack.append((heading_level, title))

        body_pos_start = match.end()
        body_pos_end = len(body) if i == len(matches) - 1 else matches[i + 1].start()

        section_body = body[body_pos_start:body_pos_end].strip()
        heading_path = tuple(t for _, t in stack)

        sections.append(Section(heading_path, section_body))

    return sections


def _calc_tokens(text: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')  # split after . ! or ? followed with whitespace


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _hard_split(text: str, tokenizer: Tokenizer, budget: int) -> list[str]:
    """
    Last resort for text with no sentence or line boundaries left. Slices the token ids into windows (budget size
    and decode back to text. Keeps chunk size limit at the cost of cutting sentences in half (hence last resort)
    """
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    return [cast(str, tokenizer.decode(ids[i : i + budget])) for i in range(0, len(ids), budget)]


def _pack_lines(lines: list[str], tokenizer: Tokenizer, budget: int) -> list[str]:
    """
    greedy fill windows with full lines up to budget tokens.
    No overlap required as lines that reach here are self contained
    """
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        # Byte pair encoding counts are not additive across joins.  Charge the join's \n into each line's budget cost
        # otherwise a window with n lines overshoots the budget by about n tokens
        line_tokens = _calc_tokens(f'\n{line}', tokenizer)
        if line_tokens > budget:
            if current:
                windows.append('\n'.join(current))
                current, current_tokens = [], 0
            windows.extend(_hard_split(line, tokenizer, budget))
            continue
        if current and current_tokens + line_tokens > budget:
            windows.append('\n'.join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += line_tokens

    if current:
        windows.append('\n'.join(current))

    return windows


@dataclass(frozen=True)
class _Unit:
    """
    Packable piece of a section body with the separator that preceded it in the source as a tag
    Packing rejoins units with their separator so a paragraph break stays a paragraph break and a
    list item stays on its own line instead of being glued onto its neighbour with a space
    """

    text: str
    sep: str  # '' for the first unit, '\n\n' between blocks, '\n' between lines, ' ' inside a line
    n_tokens: int  # text plus its separator, the join is charged up front (see _pack_lines)


def _split_units(text: str, tokenizer: Tokenizer, budget: int) -> list[_Unit]:
    """
    Cuts a body into the coarsest pieces that fits the budget:
    whole line (paragraph, list item, table caption...)
    else that line's sentences
    else raw token windows.

    Unit split is bug fix for input:
    - Alertness. You gain a bonus.
    - Dodge. You gain another bonus.

    chunked as
    a) - Alertness. You gain a bonus. - Dodge.
    b) You gain another bonus.

    Now correct
    """
    units: list[_Unit] = []

    def add(piece: str, sep: str) -> None:
        sep = '' if not units else sep  # nothing precedes the first unit
        units.append(_Unit(piece, sep, _calc_tokens(f'{sep}{piece}', tokenizer)))

    for block in re.split(r'\n{2,}', text):
        for line_index, line in enumerate(block.split('\n')):
            line = line.rstrip()  # keep leading indent, carries nested list depth
            if not line.strip():
                continue

            line_sep = '\n' if line_index else '\n\n'
            if _calc_tokens(f'{line_sep}{line}', tokenizer) <= budget:
                add(line, line_sep)
                continue

            for sentence_index, sentence in enumerate(_split_sentences(line)):
                sentence_sep = ' ' if sentence_index else line_sep
                if _calc_tokens(f'{sentence_sep}{sentence}', tokenizer) <= budget:
                    add(sentence, sentence_sep)
                    continue

                # No sentence boundary left to break on -> cut the token stream mid sentence
                for piece_index, piece in enumerate(_hard_split(sentence, tokenizer, budget)):
                    add(piece, ' ' if piece_index else sentence_sep)

    return units


def _join_units(units: list[_Unit]) -> str:
    """The first unit's separator is dropped: it belongs between this window and what came before it"""
    return units[0].text + ''.join(f'{unit.sep}{unit.text}' for unit in units[1:])


def _pack_units(units: list[_Unit], budget: int, overlap: int) -> list[str]:
    """
    Greedy fill windows up to the budget of tokens, breaking only between units.
    New windows restart with (overlap) number of tokens from the previous one.

    A unit larger than the allowance carries nothing.
    """
    windows: list[str] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in units:
        # This unit would cause a budget overflow.
        # Close the window, start a new one and carry (overlap) tokens into the new window
        if current and current_tokens + unit.n_tokens > budget:
            windows.append(_join_units(current))

            carry_forward: list[_Unit] = []
            carry_tokens = 0
            max_carry = min(overlap, budget - unit.n_tokens)
            for previous in reversed(current):
                if carry_tokens + previous.n_tokens > max_carry:
                    break
                carry_forward.insert(0, previous)
                carry_tokens += previous.n_tokens
            current = carry_forward
            current_tokens = carry_tokens

        current.append(unit)
        current_tokens += unit.n_tokens

    if current:
        windows.append(_join_units(current))

    return windows


_TABLE_SEPARATOR_RE = re.compile(r'^\|(\s*:?-{3,}:?\s*\|)+\s*$')  # the | --- | --- | divider line


def _pack_table_rows(block: str, tokenizer: Tokenizer, budget: int) -> list[str]:
    """Splits markdown pipe table. Each split has the header (col names + | --- | divider) so each split stays labelled
    header ~40 tokens, cheap to duplicate.
    Tables without a <th> row get no separator line from render_table().
    These tables have no header to repeat so they pack as plain rows
    Same fallback when the header alone consumes the whole budget
    """
    lines = block.splitlines()
    if len(lines) < 2 or not _TABLE_SEPARATOR_RE.match(lines[1].strip()):
        return _pack_lines(lines, tokenizer, budget)
    if len(lines) == 2:
        return [block]

    header = '\n'.join(lines[:2])
    row_budget = budget - _calc_tokens(f'{header}\n', tokenizer)
    if row_budget <= 0:
        return _pack_lines(lines, tokenizer, budget)

    return [f'{header}\n{window}' for window in _pack_lines(lines[2:], tokenizer, row_budget)]


def _split_body(text: str, tokenizer: Tokenizer, budget: int, overlap: int) -> list[str]:
    """
    Splits oversized section bodies into the proper path by block type on blank lines
    Tables -> row packer _pack_table_rows()
    everything else -> unit packer, which preserves the source's line and paragraph breaks
    """
    bodies: list[str] = []

    # accumulates prose blocks into a buffer rather than process them one at a time
    prose_blocks: list[str] = []

    def flush_prose() -> None:
        if prose_blocks:
            # rejoin on blank lines and resplit by _split_units which reads those breaks back off the text
            units = _split_units('\n\n'.join(prose_blocks), tokenizer, budget)
            bodies.extend(_pack_units(units, budget, overlap))
            prose_blocks.clear()

    for block in re.split(r'\n{2,}', text):
        if not block.strip():
            continue
        if block.lstrip().startswith('|'):
            #  tables are blank line delimited blocks with lines starting with "|". A table attached to a prose caption
            # would be wrongly routed to the sentence packer
            flush_prose()  # store the prose before the table
            bodies.extend(_pack_table_rows(block, tokenizer, budget))
        else:
            prose_blocks.append(block)

    flush_prose()
    return bodies


def chunk_article(article: Article, tokenizer: Tokenizer, max_tokens: int = 450, overlap: int = 50) -> list[Chunk]:
    sections = _split_by_markdown_heading(article)

    chunks: list[Chunk] = []
    idx = 0

    for section in sections:
        # A heading directly followed by a subheading or an empty article -> nothing to embed
        if not section.text:
            continue

        # Sections with no headings (preamble / no heading article) fall back to the article title as a label
        prefix = ' > '.join(section.heading_path) if section.heading_path else article.title

        full_text = f'{prefix}\n{section.text}'
        full_tokens = _calc_tokens(full_text, tokenizer)

        if full_tokens <= max_tokens:
            sized_bodies = [(full_text, full_tokens)]
        else:
            budget = max_tokens - _calc_tokens(f'{prefix}\n', tokenizer) - ceil(max_tokens * _DRIFT_MARGIN)
            if budget <= 0:
                raise ValueError(
                    f'{article.doc_id}: heading prefix {prefix!r} alone reaches max_tokens={max_tokens}, '
                    'no room left for section text'
                )
            sized_bodies = []
            for body in _split_body(section.text, tokenizer, budget, overlap):
                text = f'{prefix}\n{body}'
                sized_bodies.append((text, _calc_tokens(text, tokenizer)))

        for text, n_tokens in sized_bodies:
            chunks.append(
                Chunk(
                    chunk_id=f'{article.doc_id}#{idx:03d}',
                    doc_id=article.doc_id,
                    heading_path=list(section.heading_path),
                    text=text,
                    category=article.category,
                    n_tokens=n_tokens,
                )
            )
            idx += 1

    return chunks
