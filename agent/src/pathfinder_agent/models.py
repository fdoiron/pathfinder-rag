from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

StoppedReason = Literal['answer', 'max_iters', 'wall_clock', 'context_budget', 'fatal_tool_error', 'llm_unavailable']
ToolOutcome = Literal['ok', 'failed']

# The client holds the history and sends it back. Both the count and the fields are bounded.
#  static/index.html trims to the same MAX_HISTORY_TURNS before posting.
MAX_HISTORY_TURNS = 10


class Turn(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    answer: Annotated[str, Field(min_length=1, max_length=4000)]


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


class ExecutedToolResult(BaseModel):
    text: str
    outcome: ToolOutcome


class RunStarted(BaseModel):
    type: Literal['run_started'] = 'run_started'
    question: str


class ModelReasoning(BaseModel):
    type: Literal['model_reasoning'] = 'model_reasoning'
    text: str


class ToolStarted(BaseModel):
    type: Literal['tool_started'] = 'tool_started'
    call_id: str
    name: str
    args: dict[str, object]


class ToolFinished(BaseModel):
    type: Literal['tool_finished'] = 'tool_finished'
    call_id: str
    name: str
    outcome: ToolOutcome


class RunFinished(BaseModel):
    type: Literal['run_finished'] = 'run_finished'
    text: str
    stopped_reason: StoppedReason


class RunFailed(BaseModel):
    type: Literal['run_failed'] = 'run_failed'
    message: str


AgentEvent = RunStarted | ModelReasoning | ToolStarted | ToolFinished | RunFinished
QueueItem = AgentEvent | RunFailed
EventCallback = Callable[[AgentEvent], Awaitable[None]]
