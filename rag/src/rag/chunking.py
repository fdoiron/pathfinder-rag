import re
from dataclasses import dataclass
from typing import cast

from transformers import PreTrainedTokenizerBase

from rag.models import Article, Chunk

_HEADING_RE = re.compile(r'^(#{1,6})[ \t]+(.*)$', re.MULTILINE)


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


def _calc_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')  # split after . ! or ? folowed with whitespace


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def _hard_split(text: str, tokenizer: PreTrainedTokenizerBase, budget: int) -> list[str]:
    """
    Last resort for text with no sentence or line boundaries left. Slices the token ids into windows (budget size
    and decode back to text. Keeps chunk size limit at the cost of cutting sentences in half (hence last resort)
    """
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    return [cast(str, tokenizer.decode(ids[i : i + budget])) for i in range(0, len(ids), budget)]


def _pack_lines(lines: list[str], tokenizer: PreTrainedTokenizerBase, budget: int) -> list[str]:
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


def _pack_sentences(sentences: list[str], tokenizer: PreTrainedTokenizerBase, budget: int, overlap: int) -> list[str]:
    """
    Greedy fill windows up to the budget of tokens. New window restart with (overlap) number of tokens from the previous
    """
    windows: list[str] = []
    current: list[tuple[str, int]] = []  # (sentence, n_tokens), the carry loop never retokenizes
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _calc_tokens(f' {sentence}', tokenizer)  # charge the joining space (see _pack_lines)

        # If a sentence is bigger than budget drop to packing the sentence's lines instead
        if sentence_tokens > budget:
            if current:
                windows.append(' '.join(s for s, _ in current))
                current = []
                current_tokens = 0
            lines = [line for line in sentence.splitlines() if line.strip()]
            windows.extend(_pack_lines(lines, tokenizer, budget))
            continue

        # This sentence would cause a budget overflow.
        # Close the window, start a new one and carry (overlap) tokens into the new window
        if current_tokens + sentence_tokens > budget:
            windows.append(' '.join(s for s, _ in current))

            carry_forward: list[tuple[str, int]] = []
            carry_tokens = 0
            max_carry = min(overlap, budget - sentence_tokens)
            for previous, previous_tokens in reversed(current):
                if carry_tokens + previous_tokens > max_carry:
                    break
                carry_forward.insert(0, (previous, previous_tokens))
                carry_tokens += previous_tokens
            current = carry_forward
            current_tokens = carry_tokens

        current.append((sentence, sentence_tokens))
        current_tokens += sentence_tokens

    if current:
        windows.append(' '.join(s for s, _ in current))

    return windows


_TABLE_SEPARATOR_RE = re.compile(r'^\|(\s*:?-{3,}:?\s*\|)+\s*$')  # the | --- | --- | divider line


def _pack_table_rows(block: str, tokenizer: PreTrainedTokenizerBase, budget: int) -> list[str]:
    """Splits markdown pipe table. Each split has the header (col names + | --- | divider) so each split stays labelled
    header ~40 tokens, cheap to duplibcate.
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


def _split_body(text: str, tokenizer: PreTrainedTokenizerBase, budget: int, overlap: int) -> list[str]:
    """
    Splits oversized section bodies into the proper path by block type on blank lines
    Tables -> row packer _pack_table_rows()
    everything else -> sentence packer _pack_sentences()
    """
    bodies: list[str] = []

    # accumulates prose blocks into a buffer rather than process them one at a time
    prose_blocks: list[str] = []

    def flush_prose() -> None:
        if prose_blocks:
            sentences = _split_sentences('\n\n'.join(prose_blocks))
            bodies.extend(_pack_sentences(sentences, tokenizer, budget, overlap))
            prose_blocks.clear()

    for block in re.split(r'\n{2,}', text):
        if not block.strip():
            continue
        if block.lstrip().startswith('|'):
            #  tables are blank line delimited blocks with lines starting with "|". A tabled attached to a prose caption
            # would be wrongly routed to the sentence packer
            flush_prose()  # store the prose before the table
            bodies.extend(_pack_table_rows(block, tokenizer, budget))
        else:
            prose_blocks.append(block)

    flush_prose()
    return bodies


def chunk_article(
    article: Article, tokenizer: PreTrainedTokenizerBase, max_tokens: int = 450, overlap: int = 50
) -> list[Chunk]:
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
            budget = max_tokens - _calc_tokens(f'{prefix}\n', tokenizer)
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
