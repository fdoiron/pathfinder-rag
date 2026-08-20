import json
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from pathfinder_agent.config import Settings
from pathfinder_agent.models import (
    MAX_HISTORY_TURNS,
    AgentEvent,
    AgentResult,
    EventCallback,
    RunFinished,
    RunStarted,
    ToolCallRecord,
    ToolFinished,
    ToolStarted,
    Turn,
)
from pathfinder_agent.server import AskRequest, app, ask

client = TestClient(app)

QUESTION = 'How does grappling work?'
HOP: list[AgentEvent] = [
    RunStarted(question=QUESTION),
    ToolStarted(call_id='call-1', name='rag_search', args={'query': 'grapple'}),
    ToolFinished(call_id='call-1', name='rag_search', outcome='ok'),
    RunFinished(text='You make a CMB check.', stopped_reason='answer'),
]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# helpers
def _stub_run_question(
    monkeypatch: pytest.MonkeyPatch,
    events: list[AgentEvent] | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Replace run_question with one that emits `events` then raises `error`, recording each call."""
    calls: list[dict[str, Any]] = []

    async def fake(
        question: str,
        settings: Settings,  # noqa: ARG001
        history: list[Turn] | None = None,
        on_event: EventCallback | None = None,
    ) -> AgentResult:
        calls.append({'question': question, 'history': history})
        for event in events if events is not None else HOP:
            if on_event:
                await on_event(event)
        if error:
            raise error
        return AgentResult(
            text='You make a CMB check.',
            tool_calls=[ToolCallRecord(name='rag_search', args={'query': 'grapple'})],
            stopped_reason='answer',
        )

    monkeypatch.setattr('pathfinder_agent.server.run_question', fake)
    return calls


async def _drain(request: AskRequest) -> list[dict[str, str]]:
    """The frames the SSE generator yields, without going through a socket."""
    response = await ask(request, Settings())
    return [cast(dict[str, str], frame) async for frame in response.body_iterator]


# the event stream
@pytest.mark.anyio
async def test_ask_streams_every_event_the_run_emitted_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch)

    frames = await _drain(AskRequest(question=QUESTION))

    assert [frame['event'] for frame in frames] == ['run_started', 'tool_started', 'tool_finished', 'run_finished']


@pytest.mark.anyio
async def test_ask_names_each_frame_after_the_type_in_its_own_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch)

    frames = await _drain(AskRequest(question=QUESTION))

    assert all(json.loads(frame['data'])['type'] == frame['event'] for frame in frames)


@pytest.mark.anyio
async def test_ask_serialises_an_event_with_every_field_the_page_draws_from(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch)

    frames = await _drain(AskRequest(question=QUESTION))

    assert json.loads(frames[1]['data']) == {
        'type': 'tool_started',
        'call_id': 'call-1',
        'name': 'rag_search',
        'args': {'query': 'grapple'},
    }


# failure
@pytest.mark.anyio
async def test_ask_closes_a_run_that_raised_with_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch, events=[], error=RuntimeError('the MCP server died mid run'))

    frames = await _drain(AskRequest(question=QUESTION))

    assert [frame['event'] for frame in frames] == ['run_failed']


@pytest.mark.anyio
async def test_ask_does_not_put_the_underlying_error_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch, events=[], error=RuntimeError('postgres://user:hunter2@db/rag'))

    frames = await _drain(AskRequest(question=QUESTION))

    assert 'hunter2' not in frames[0]['data']


@pytest.mark.anyio
async def test_ask_keeps_the_events_a_run_emitted_before_it_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_question(monkeypatch, events=HOP[:2], error=RuntimeError('died after the first hop'))

    frames = await _drain(AskRequest(question=QUESTION))

    assert [frame['event'] for frame in frames] == ['run_started', 'tool_started', 'run_failed']


# the request
@pytest.mark.anyio
async def test_ask_passes_the_history_through_to_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run_question(monkeypatch)
    history = [Turn(question='What is a red dragon?', answer='A chromatic dragon.')]

    await _drain(AskRequest(question="What's its AC?", history=history))

    assert calls[0] == {'question': "What's its AC?", 'history': history}


@pytest.mark.anyio
async def test_ask_runs_a_question_with_no_history_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run_question(monkeypatch)

    await _drain(AskRequest(question=QUESTION))

    assert calls[0]['history'] == []


@pytest.mark.parametrize(
    ('payload', 'loc'),
    [
        pytest.param({'question': ''}, ['body', 'question'], id='empty question'),
        pytest.param({'question': 'x' * 1001}, ['body', 'question'], id='long question'),
        pytest.param(
            {'question': 'q', 'history': [{'question': 'q', 'answer': 'a'}] * (MAX_HISTORY_TURNS + 1)},
            ['body', 'history'],
            id='too many turns',
        ),
        pytest.param(
            {'question': 'q', 'history': [{'question': 'q', 'answer': 'x' * 4001}]},
            ['body', 'history', 0, 'answer'],
            id='long answer',
        ),
        pytest.param(
            {'question': 'q', 'history': [{'question': 'q', 'answer': ''}]},
            ['body', 'history', 0, 'answer'],
            id='empty answer',
        ),
    ],
)
def test_ask_rejects_a_request_outside_the_bounds_it_declares(payload: dict[str, Any], loc: list[Any]) -> None:
    response = client.post('/ask', json=payload)

    assert response.status_code == 422
    assert response.json()['detail'][0]['loc'] == loc


# the mount
def test_the_page_is_served_from_the_root_behind_the_endpoint() -> None:
    response = client.get('/')

    assert response.status_code == 200
    assert '/ask' in response.text
