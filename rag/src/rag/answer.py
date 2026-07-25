import re

from openai import OpenAI
from pydantic import BaseModel

from rag.config import Settings
from rag.retrieval import Retriever


def make_llm_client(settings: Settings) -> OpenAI:
    # Ollama ignores the API key but the SDK insists on one
    return OpenAI(base_url=settings.llm_base_url, api_key='ollama', timeout=settings.llm_timeout)


class Citation(BaseModel):
    n: int
    title: str
    heading_path: list[str]
    url: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]


_CITATION_RE = re.compile(r'\[(\d+)\]')
_NO_COVERAGE_REPLY = "The retrieved excerpts don't cover this."


def answer_question(
    question: str,
    retriever: Retriever,
    client: OpenAI,
    settings: Settings,
    k: int | None = None,
    category: str | None = None,
) -> Answer:
    hits = retriever.search(question, k=k if k is not None else settings.ask_k, category=category)
    if not hits:
        return Answer(text=_NO_COVERAGE_REPLY, citations=[])

    excerpts = '\n\n'.join(
        f'[{n}] {hit.title} — {" > ".join(hit.heading_path)}\n{hit.text}' for n, hit in enumerate(hits, start=1)
    )
    template = settings.ask_prompt_path.read_text(encoding='utf-8')
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{'role': 'user', 'content': template.format(excerpts=excerpts, question=question)}],
    )
    text = response.choices[0].message.content or ''

    cited = sorted({int(m) for m in _CITATION_RE.findall(text)})
    citations = [
        Citation(n=n, title=hits[n - 1].title, heading_path=hits[n - 1].heading_path, url=str(hits[n - 1].url))
        for n in cited
        if 1 <= n <= len(hits)  # don't resolve hallucinated citations
    ]
    return Answer(text=text, citations=citations)
