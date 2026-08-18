import asyncio
from typing import Any

import httpx2
import pytest
from mcp.shared.exceptions import MCPError
from typer.testing import CliRunner

from pathfinder_agent.cli import MCPUnavailableError, app, leaf_causes, run_question
from pathfinder_agent.config import Settings
from pathfinder_agent.models import AgentResult, ToolCallRecord

runner = CliRunner()


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# helpers
def _task_group(*causes: BaseException) -> BaseExceptionGroup[Any]:
    """what anyio raises. collapsed to ExceptionGroup unless it is a bare BaseException"""
    return BaseExceptionGroup('unhandled errors in a TaskGroup', list(causes))


def _result(stopped_reason: str, text: str = 'You make a CMB check.') -> AgentResult:
    return AgentResult(
        text=text,
        tool_calls=[ToolCallRecord(name='rag_search', args={'query': 'grapple'})],
        stopped_reason=stopped_reason,
    )


def _stub_session(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def explode(settings: Settings) -> Any:  # noqa: ARG001
        raise error

    monkeypatch.setattr('pathfinder_agent.cli.open_mcp_session', explode)


def _stub_run_question(monkeypatch: pytest.MonkeyPatch, outcome: AgentResult | Exception) -> None:
    async def fake(question: str, settings: Settings) -> AgentResult:  # noqa: ARG001
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr('pathfinder_agent.cli.run_question', fake)


# leaf_causes
# ExceptionGroup('unhandled errors in a TaskGroup'): wrapper
# ─ ExceptionGroup('unhandled errors in a TaskGroup'): wrapper
#    ─ MCPError('SSE stream ended without a response'): leaf


def test_leaf_causes_unwraps_one_task_group() -> None:
    group = _task_group(httpx2.ConnectError('All connection attempts failed'))

    assert [str(e) for e in leaf_causes(group)] == ['All connection attempts failed']


def test_leaf_causes_unwraps_the_nested_group_a_mid_run_death_raises() -> None:
    group = _task_group(_task_group(MCPError(-32000, 'SSE stream ended without a response')))

    assert [str(e) for e in leaf_causes(group)] == ['SSE stream ended without a response']


def test_leaf_causes_keeps_every_cause() -> None:
    group = _task_group(ValueError('first'), RuntimeError('second'))

    assert [str(e) for e in leaf_causes(group)] == ['first', 'second']


def test_leaf_causes_passes_a_plain_exception_through() -> None:
    error = RuntimeError('not a group')

    assert leaf_causes(error) == [error]


# run_question
@pytest.mark.anyio
@pytest.mark.parametrize(
    ('group', 'expected'),
    [
        pytest.param(
            _task_group(httpx2.ConnectError('All connection attempts failed')),
            'ConnectError: All connection attempts failed',
            id='never started',
        ),
        pytest.param(
            _task_group(_task_group(MCPError(-32000, 'SSE stream ended without a response'))),
            'MCPError: SSE stream ended without a response',
            id='died mid run',
        ),
    ],
)
async def test_run_question_translates_a_dead_transport(
    monkeypatch: pytest.MonkeyPatch, group: BaseExceptionGroup[Any], expected: str
) -> None:
    _stub_session(monkeypatch, group)
    settings = Settings(mcp_server_url='http://localhost:8000/mcp')

    with pytest.raises(MCPUnavailableError) as excinfo:
        await run_question('How does grappling work?', settings)

    assert 'http://localhost:8000/mcp' in str(excinfo.value)
    assert expected in str(excinfo.value)


@pytest.mark.anyio
async def test_run_question_lets_a_real_bug_keep_its_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_session(monkeypatch, AttributeError("'NoneType' object has no attribute 'search'"))

    with pytest.raises(AttributeError):  # not reported as unreachable server
        await run_question('How does grappling work?', Settings())


def test_run_question_does_not_swallow_a_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_session(monkeypatch, _task_group(KeyboardInterrupt()))

    with pytest.raises(BaseExceptionGroup):
        asyncio.run(run_question('How does grappling work?', Settings()))


# ask exit codes
def test_ask_exits_zero_on_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch, _result('answer'))

    result = runner.invoke(app, ['ask', 'How does grappling work?'])

    assert result.exit_code == 0
    assert 'You make a CMB check.' in result.stdout
    assert "rag_search {'query': 'grapple'}" in result.stdout


@pytest.mark.parametrize('stopped_reason', ['max_iters', 'wall_clock', 'context_budget'])
def test_ask_exits_zero_when_a_partial_answer_came_back(monkeypatch: pytest.MonkeyPatch, stopped_reason: str) -> None:
    _stub_run_question(monkeypatch, _result(stopped_reason))

    result = runner.invoke(app, ['ask', 'How does grappling work?'])

    assert result.exit_code == 0
    assert f'stopped: {stopped_reason}' in result.stderr


@pytest.mark.parametrize('stopped_reason', ['llm_unavailable', 'fatal_tool_error'])
def test_ask_exits_one_when_the_question_went_unanswered(monkeypatch: pytest.MonkeyPatch, stopped_reason: str) -> None:
    _stub_run_question(monkeypatch, _result(stopped_reason, text='The LLM could not be reached.'))

    result = runner.invoke(app, ['ask', 'How does grappling work?'])

    assert result.exit_code == 1
    assert 'The LLM could not be reached.' in result.stdout  # the reason prints before the exit
    assert f'stopped: {stopped_reason}' in result.stderr


def test_ask_reports_an_unreachable_server_without_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch, MCPUnavailableError('The MCP server at http://localhost:8000/mcp is not reachable'))

    result = runner.invoke(app, ['ask', 'How does grappling work?'])

    assert result.exit_code == 1
    assert 'http://localhost:8000/mcp is not reachable' in result.stderr
    assert 'Traceback' not in result.stderr
    assert isinstance(result.exception, SystemExit)  #  not the error escaping the command
