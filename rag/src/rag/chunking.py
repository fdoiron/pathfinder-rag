import re
from dataclasses import dataclass

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

    # Article starts with a no-heading preamble section
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


def chunk_article(page: Article) -> list[Chunk]:
    raise NotImplementedError
