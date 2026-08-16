import contextlib
from collections.abc import Iterator

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, NotFoundError

from pathfinder_agent.config import Settings


def make_llm_client(settings: Settings) -> AsyncOpenAI:
    # Ollama ignores the API key but the SDK insists on one.
    # max_retries=0: run_agent controls the retry policy. The SDK's default (2) would take 3 hops before raising.
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key='ollama',
        timeout=settings.agent_hop_timeout,
        max_retries=0,
    )


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint is unreachable or the configured model is not served there."""


class LLMTimeoutError(LLMUnavailableError):
    """The LLM did not answer within the hop budget. Retryable."""


@contextlib.contextmanager
def translate_llm_errors(settings: Settings) -> Iterator[None]:
    """Raise LLMUnavailableError for any SDK transport or status error from the wrapped call."""
    try:
        yield
    except APITimeoutError as e:  # subclass of APIConnectionError has to be caught first
        raise LLMTimeoutError(
            f'{settings.llm_model!r} at {settings.llm_base_url} did not answer within '
            f'{settings.agent_hop_timeout:.0f}s.'
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
