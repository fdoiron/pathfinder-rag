import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError

from pathfinder_agent.config import Settings
from pathfinder_agent.llm import LLMTimeoutError, LLMUnavailableError, make_llm_client, translate_llm_errors

REQUEST = httpx.Request('POST', 'http://llm.test/v1/chat/completions')


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_base_url='http://llm.test/v1', llm_model='qwen3:14b', agent_hop_timeout=30.0)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=REQUEST)


def test_timeout_maps_to_the_retryable_subclass(settings: Settings) -> None:
    with pytest.raises(LLMTimeoutError) as excinfo, translate_llm_errors(settings):
        raise APITimeoutError(request=REQUEST)

    assert "'qwen3:14b'" in str(excinfo.value)
    assert '30s' in str(excinfo.value)


def test_connection_error_does_not_masquerade_as_a_timeout(settings: Settings) -> None:
    with pytest.raises(LLMUnavailableError) as excinfo, translate_llm_errors(settings):
        raise APIConnectionError(request=REQUEST)

    assert not isinstance(excinfo.value, LLMTimeoutError)
    assert 'http://llm.test/v1' in str(excinfo.value)


def test_missing_model_is_reported_as_a_model_problem(settings: Settings) -> None:
    with pytest.raises(LLMUnavailableError) as excinfo, translate_llm_errors(settings):
        raise NotFoundError('model not found', response=_response(404), body=None)

    assert 'does not serve model' in str(excinfo.value)
    assert "'qwen3:14b'" in str(excinfo.value)


def test_other_status_errors_carry_the_status_code(settings: Settings) -> None:
    with pytest.raises(LLMUnavailableError) as excinfo, translate_llm_errors(settings):
        raise APIStatusError('overloaded', response=_response(503), body=None)

    assert 'HTTP 503' in str(excinfo.value)
    assert 'overloaded' in str(excinfo.value)


def test_unrelated_exceptions_pass_through(settings: Settings) -> None:
    with pytest.raises(ZeroDivisionError), translate_llm_errors(settings):
        raise ZeroDivisionError


def test_client_leaves_retry_and_timeout_policy_to_the_agent(settings: Settings) -> None:
    client = make_llm_client(settings)

    assert client.max_retries == 0
    assert client.timeout == settings.agent_hop_timeout
    assert str(client.base_url) == 'http://llm.test/v1/'  # the SDK appends the trailing slash
