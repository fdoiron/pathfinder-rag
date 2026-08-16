import asyncio
import importlib.resources
import logging
from typing import Annotated

import typer

from pathfinder_agent.agent import AgentResult, open_mcp_session, run_agent
from pathfinder_agent.config import Settings, get_settings
from pathfinder_agent.llm import make_llm_client

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx2').setLevel(logging.WARNING)

# stopped_reason values that left the question unanswered so the CLI exits without zero
FAILED_REASONS = frozenset({'llm_unavailable', 'fatal_tool_error'})


class MCPUnavailableError(RuntimeError):
    """The MCP server was unreachable at startup or stopped answering during a run."""


@app.callback()
def _callback() -> None:
    """Pathfinder 1e RAG agent CLI."""


def load_system_prompt(settings: Settings) -> str:
    if settings.agent_system_prompt_path is not None:
        return settings.agent_system_prompt_path.read_text(encoding='utf-8')
    return (
        importlib.resources.files('pathfinder_agent')
        .joinpath('prompts', 'agent_system.txt')
        .read_text(encoding='utf-8')
    )


def leaf_causes(exc: BaseException) -> list[BaseException]:
    """Collapse the transport's nested task group ExceptionGroups down to the exceptions that failed."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in leaf_causes(sub)]
    return [exc]


async def run_question(question: str, settings: Settings) -> AgentResult:
    llm_client = make_llm_client(settings=settings)
    system_prompt = load_system_prompt(settings)

    try:
        async with open_mcp_session(settings) as mcp_session:
            return await run_agent(
                question=question,
                mcp_session=mcp_session,
                llm_client=llm_client,
                settings=settings,
                system_prompt=system_prompt,
            )
    except ExceptionGroup as eg:
        causes = '; '.join(f'{type(e).__name__}: {e}' for e in leaf_causes(eg))
        raise MCPUnavailableError(f'The MCP server at {settings.mcp_server_url} is not reachable: {causes}') from eg


@app.command()
def ask(question: Annotated[str, typer.Argument(help='a rules question')]) -> None:
    """Answer a Pathfinder 1e rules question using the RAG MCP server."""
    settings = get_settings()
    try:
        result = asyncio.run(run_question(question, settings))
    except MCPUnavailableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    typer.echo(result.text)
    typer.echo()
    for call in result.tool_calls:
        typer.echo(f'{call.name} {call.args}')
    if result.stopped_reason != 'answer':
        typer.echo(f'stopped: {result.stopped_reason}', err=True)
    if result.stopped_reason in FAILED_REASONS:
        raise typer.Exit(1)


if __name__ == '__main__':
    app()
