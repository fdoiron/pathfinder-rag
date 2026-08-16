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


async def run_question(question: str, settings: Settings) -> AgentResult:
    # TODO: a mid run MCP server death raises an ExceptionGroup from the transport's background GET
    # stream, inside this task group and above run_agent, so execute_tool's retry never sees it.
    # Catch it here and exit with a message instead of a traceback.
    llm_client = make_llm_client(settings=settings)
    system_prompt = load_system_prompt(settings)

    async with open_mcp_session(settings) as mcp_session:
        return await run_agent(
            question=question,
            mcp_session=mcp_session,
            llm_client=llm_client,
            settings=settings,
            system_prompt=system_prompt,
        )


@app.command()
def ask(question: Annotated[str, typer.Argument(help='a rules question')]) -> None:
    """Answer a Pathfinder 1e rules question using the RAG MCP server."""
    settings = get_settings()
    result = asyncio.run(run_question(question, settings))

    typer.echo(result.text)
    typer.echo()
    for call in result.tool_calls:
        typer.echo(f'{call.name} {call.args}')
    if result.stopped_reason != 'answer':
        typer.echo(f'stopped: {result.stopped_reason}', err=True)


if __name__ == '__main__':
    app()
