import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)
from opentelemetry import trace

from pathfinder_agent.config import Settings
from pathfinder_agent.llm import LLMUnavailableError, translate_llm_errors
from pathfinder_agent.models import AgentResult, ClassifiedToolResult, ToolCallRecord
from pathfinder_agent.telemetry import configure_telemetry

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

NO_COVERAGE_REPLY = "The searches I ran don't cover this."

REPHRASE_INSTRUCTION = (
    'Correct the arguments and call the tool again; run rag_search first if you need a valid chunk_id.'
)

FORCED_ANSWER_INSTRUCTION = (
    'No more tool calls are available. Answer using ONLY the text inside the <tool_result> tags above, citing '
    'the URL for each claim. If those results do not answer the question, reply exactly:\n'
    f'{NO_COVERAGE_REPLY}'
)


class RetryableToolError(Exception):
    pass


class WallClockExpired(Exception):
    pass


async def call_mcp_tool(session: ClientSession, name: str, args: dict[str, Any]) -> ClassifiedToolResult:
    result = await session.call_tool(name, arguments=args)
    content = result.content[0] if result.content else None
    text = content.text if isinstance(content, TextContent) else ''
    error_category = (result.structured_content or {}).get('error_category')
    if not error_category and result.is_error:
        if '[retryable]' in text:
            error_category = 'retryable'
        elif '[rephrase]' in text:
            error_category = 'rephrase'
        elif '[fatal]' in text:
            error_category = 'fatal'
    return ClassifiedToolResult(text=text, error_category=error_category)


async def execute_tool(
    tool_call: ChatCompletionMessageFunctionToolCall,
    args: dict[str, Any],
    mcp_session: ClientSession,
    settings: Settings,
    deadline: float,
) -> str:
    delay = settings.agent_retry_backoff_base
    attempts = settings.agent_max_tool_attempts
    attempt = 0

    with tracer.start_as_current_span('execute_tool') as span:
        span.set_attribute('tool.name', tool_call.function.name)
        span.set_attribute('tool.args', json.dumps(args))
        while attempt < attempts:
            span.set_attribute('tool.attempts', attempt + 1)
            time_left = deadline - time.monotonic()
            if time_left <= 0:
                raise WallClockExpired
            per_call_timeout = min(settings.agent_hop_timeout, time_left)

            try:
                result = await asyncio.wait_for(
                    call_mcp_tool(mcp_session, tool_call.function.name, args), timeout=per_call_timeout
                )
                if result.error_category:
                    span.set_attribute('tool.error_category', result.error_category)

                if result.error_category == 'rephrase':
                    return f'{result.text}\n{REPHRASE_INSTRUCTION}'
                elif result.error_category == 'retryable':
                    raise RetryableToolError(result.text)
                elif result.error_category:
                    raise RuntimeError(f'Fatal tool error: {result.text}')
                else:
                    return result.text
            except RuntimeError as e:
                span.record_exception(e)
                raise
            except Exception as e:
                attempt += 1
                if attempt < attempts:
                    await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                    delay = delay * 2
                    logger.warning(f'Error executing tool: {e}. Attempt {attempt} of {attempts}.')
                else:
                    span.record_exception(e)
                    logger.error('Max attempts reached. Failing.')
                    raise

    raise AssertionError('unreachable: loop always returns or raises before exhausting attempt < attempts')


async def translate_mcp_tools(mcp_session: ClientSession) -> list[ChatCompletionFunctionToolParam]:
    result = await mcp_session.list_tools()
    return [
        {
            'type': 'function',
            'function': {
                'name': t.name,
                'description': t.description or '',
                'parameters': t.input_schema,
            },
        }
        for t in result.tools
    ]


def wrap_tool_result(text: str) -> str:
    # TODO: a per run nonce in the tag would make escaping unnecessary. Needs agent_system.txt, which names the tag in
    # prose, to become a template first.
    safe_text = re.sub(r'</\s*tool_result\s*>', '<\u200b/tool_result>', text, flags=re.IGNORECASE)
    return f'<tool_result>\n{safe_text}\n</tool_result>'


async def run_agent(
    question: str, mcp_session: ClientSession, llm_client: AsyncOpenAI, settings: Settings, system_prompt: str
) -> AgentResult:

    messages: list[ChatCompletionMessageParam] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question},
    ]
    deadline = time.monotonic() + settings.agent_wall_clock_timeout
    max_iters = settings.agent_max_iters
    attempt = 0
    tool_token_count = 0
    tool_calls: list[ToolCallRecord] = []
    budget_exceeded = False
    answer_forced = False

    with tracer.start_as_current_span('run_agent') as span:
        span.set_attribute('agent.model', settings.llm_model)
        span.set_attribute('agent.max_iters', max_iters)

        def finish(result: AgentResult) -> AgentResult:
            span.set_attribute('agent.stopped_reason', result.stopped_reason)
            span.set_attribute('agent.iterations', attempt)
            span.set_attribute('agent.tool_calls', len(result.tool_calls))
            return result

        available_tools = await translate_mcp_tools(mcp_session=mcp_session)
        while attempt < max_iters:
            attempt += 1
            if time.monotonic() > deadline:
                return finish(
                    AgentResult(
                        text='The run took too long and had to stop before finishing.',
                        tool_calls=tool_calls,
                        stopped_reason='wall_clock',
                    )
                )

            budget_exceeded = tool_token_count > settings.agent_tool_result_token_budget
            if not answer_forced and (budget_exceeded or attempt == max_iters):
                answer_forced = True
                available_tools = []
                messages.append({'role': 'system', 'content': FORCED_ANSWER_INSTRUCTION})

            try:
                with translate_llm_errors(settings):
                    response = await llm_client.chat.completions.create(
                        model=settings.llm_model, messages=messages, tools=available_tools
                    )

            except LLMUnavailableError as e:
                logger.error(f'LLM unavailable: {e}')
                span.record_exception(e)
                return finish(
                    AgentResult(
                        text=f'The LLM could not be reached: {e}',
                        tool_calls=tool_calls,
                        stopped_reason='llm_unavailable',
                    )
                )

            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({'role': 'assistant', 'content': msg.content})
                stopped_reason = 'context_budget' if budget_exceeded else 'max_iters' if answer_forced else 'answer'

                return finish(AgentResult(text=msg.content or '', tool_calls=tool_calls, stopped_reason=stopped_reason))

            tool_call_params = cast(
                list[ChatCompletionMessageFunctionToolCallParam | ChatCompletionMessageCustomToolCallParam],
                [tc.model_dump() for tc in msg.tool_calls],
            )
            messages.append({'role': 'assistant', 'content': msg.content, 'tool_calls': tool_call_params})

            for call in msg.tool_calls:
                try:
                    if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                        raise RuntimeError(f'Unsupported tool call type: {type(call).__name__}')
                    args = json.loads(call.function.arguments)
                    tool_calls.append(ToolCallRecord(name=call.function.name, args=args))
                    text = await execute_tool(call, args, mcp_session, settings, deadline=deadline)
                except json.JSONDecodeError as e:
                    text = f'[rephrase] Invalid JSON arguments: {e}'
                except WallClockExpired:
                    return finish(
                        AgentResult(
                            text='The run took too long and had to stop before finishing.',
                            tool_calls=tool_calls,
                            stopped_reason='wall_clock',
                        )
                    )
                except Exception as e:
                    logger.error(f'Fatal tool error: {e}')
                    span.record_exception(e)
                    return finish(
                        AgentResult(
                            text=f'The search tool failed and the run could not continue: {e}',
                            tool_calls=tool_calls,
                            stopped_reason='fatal_tool_error',
                        )
                    )

                tool_token_count += len(text) // 4  # Approximation to avoid loading embedder
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': call.id,
                        'content': wrap_tool_result(text),
                    }
                )

        return finish(AgentResult(text=msg.content or '', tool_calls=tool_calls, stopped_reason='max_iters'))


@asynccontextmanager
async def open_mcp_session(settings: Settings) -> AsyncGenerator[ClientSession]:
    configure_telemetry(settings, 'rag-agent-client')
    token = settings.mcp_auth_token
    headers = {'Authorization': f'Bearer {token.get_secret_value()}'} if token else {}

    async with (
        create_mcp_http_client(headers) as http_client,
        streamable_http_client(settings.mcp_server_url, http_client=http_client) as (read, write),
        ClientSession(read, write) as mcp_session,
    ):
        await mcp_session.initialize()

        yield mcp_session
