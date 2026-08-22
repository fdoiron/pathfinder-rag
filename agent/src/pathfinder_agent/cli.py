import asyncio
import logging
from typing import Annotated

import typer

from pathfinder_agent.config import get_settings
from pathfinder_agent.runner import MCPUnavailableError, run_question

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx2').setLevel(logging.WARNING)

# stopped_reason values that left the question unanswered so the CLI exits without zero
FAILED_REASONS = frozenset({'llm_unavailable', 'fatal_tool_error'})


@app.callback()
def _callback() -> None:
    """Pathfinder 1e RAG agent CLI."""


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
