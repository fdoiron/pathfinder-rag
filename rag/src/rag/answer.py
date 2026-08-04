import importlib.resources
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError, OpenAI
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.models import ChunkHit
from rag.retrieval import Retriever, SearchMethod


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint is unreachable or too slow, or the configured model is not served there."""


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
    invented_citations: list[int] = Field(default_factory=list)  # cited numbers outside the retrieved set


_CITATION_RE = re.compile(r'\[(\d+)\]')
NO_COVERAGE_REPLY = "The retrieved excerpts don't cover this."


def answer_question(
    question: str,
    retriever: Retriever,
    client: OpenAI,
    settings: Settings,
    k: int | None = None,
    category: str | None = None,
    method: SearchMethod = 'hybrid',
    rerank: bool = False,
) -> tuple[Answer, list[ChunkHit]]:
    hits = retriever.search(
        question, k=k if k is not None else settings.ask_k, category=category, method=method, rerank=rerank
    )
    if not hits:
        return Answer(text=NO_COVERAGE_REPLY, citations=[]), hits

    excerpts = '\n\n'.join(
        f'[{n}] {hit.title} — {" > ".join(hit.heading_path)}\n{hit.text}' for n, hit in enumerate(hits, start=1)
    )
    if settings.ask_prompt_path is not None:
        template = settings.ask_prompt_path.read_text(encoding='utf-8')
    else:
        template = importlib.resources.files('rag').joinpath('prompts', 'ask.txt').read_text(encoding='utf-8')
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{'role': 'user', 'content': template.format(excerpts=excerpts, question=question)}],
        )
    except APITimeoutError as e:  # subclass of APIConnectionError has to be caught first
        raise LLMUnavailableError(
            f'{settings.llm_model!r} at {settings.llm_base_url} did not answer within {settings.llm_timeout:.0f}s. '
        ) from e
    except APIConnectionError as e:
        raise LLMUnavailableError(f'No LLM server reachable at {settings.llm_base_url}.') from e
    except NotFoundError as e:
        raise LLMUnavailableError(
            f'The server at {settings.llm_base_url} does not serve model {settings.llm_model!r}.'
        ) from e
    except APIStatusError as e:
        raise LLMUnavailableError(
            f'The server at {settings.llm_base_url} returned HTTP {e.status_code} for model '
            f'{settings.llm_model!r}: {e.message}'
        ) from e
    text = response.choices[0].message.content or ''

    cited_raw = sorted({int(m) for m in _CITATION_RE.findall(text)})
    invented = [n for n in cited_raw if not (1 <= n <= len(hits))]
    citations = [
        Citation(n=n, title=hits[n - 1].title, heading_path=hits[n - 1].heading_path, url=str(hits[n - 1].url))
        for n in cited_raw
        if 1 <= n <= len(hits)  # don't resolve invented citations
    ]
    return Answer(text=text, citations=citations, invented_citations=invented), hits
