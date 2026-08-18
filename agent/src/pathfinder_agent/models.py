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
