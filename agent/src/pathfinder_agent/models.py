from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel

StoppedReason = Literal['answer', 'max_iters', 'wall_clock', 'context_budget', 'fatal_tool_error', 'llm_unavailable']


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]


class AgentResult(BaseModel):
    text: str
    tool_calls: list[ToolCallRecord]
    stopped_reason: StoppedReason


class ClassifiedToolResult(BaseModel):
    text: str
    error_category: Literal['retryable', 'rephrase', 'fatal'] | None


class RunStarted(BaseModel):
    type: Literal['run_started'] = 'run_started'
    question: str


class ToolStarted(BaseModel):
    type: Literal['tool_started'] = 'tool_started'
    call_id: str
    name: str
    args: dict[str, object]


class ToolFinished(BaseModel):
    type: Literal['tool_finished'] = 'tool_finished'
    call_id: str
    name: str


class RunFinished(BaseModel):
    type: Literal['run_finished'] = 'run_finished'
    text: str
    stopped_reason: StoppedReason


class RunFailed(BaseModel):
    type: Literal['run_failed'] = 'run_failed'
    message: str


AgentEvent = RunStarted | ToolStarted | ToolFinished | RunFinished
QueueItem = AgentEvent | RunFailed
EventCallback = Callable[[AgentEvent], Awaitable[None]]
