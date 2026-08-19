import importlib.resources

from pathfinder_agent.agent import open_mcp_session, run_agent
from pathfinder_agent.config import Settings
from pathfinder_agent.llm import make_llm_client
from pathfinder_agent.models import AgentResult, EventCallback


class MCPUnavailableError(RuntimeError):
    """The MCP server was unreachable at startup or stopped answering during a run."""


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


async def run_question(question: str, settings: Settings, on_event: EventCallback | None = None) -> AgentResult:
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
                on_event=on_event,
            )
    except ExceptionGroup as eg:
        causes = '; '.join(f'{type(e).__name__}: {e}' for e in leaf_causes(eg))
        raise MCPUnavailableError(f'The MCP server at {settings.mcp_server_url} is not reachable: {causes}') from eg
