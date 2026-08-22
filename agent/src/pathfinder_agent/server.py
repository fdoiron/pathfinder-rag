import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pathfinder_agent.config import Settings, get_settings
from pathfinder_agent.models import MAX_HISTORY_TURNS, AgentEvent, QueueItem, RunFailed, Turn
from pathfinder_agent.runner import run_question

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx2').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_log_lock = asyncio.Lock()


async def log_event(settings: Settings, run_id: str, event: QueueItem) -> None:
    """Append one event to the interaction log, the same serialization the page receives."""
    if settings.interaction_log_path is None:
        return
    line = json.dumps(
        {
            'ts': datetime.now(UTC).isoformat(timespec='seconds'),
            'run': run_id,
            'event': json.loads(event.model_dump_json()),
        },
        ensure_ascii=False,
    )
    async with _log_lock:  # two questions at once would otherwise interleave mid line
        with settings.interaction_log_path.open('a', encoding='utf-8') as log:
            log.write(line + '\n')


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    history: Annotated[list[Turn], Field(max_length=MAX_HISTORY_TURNS)] = []


@app.post('/ask')
async def ask(ask_request: AskRequest, settings: Annotated[Settings, Depends(get_settings)]) -> EventSourceResponse:
    queue: asyncio.Queue[QueueItem | None] = asyncio.Queue()
    run_id = uuid.uuid4().hex[:8]

    async def on_event(event: AgentEvent) -> None:
        await log_event(settings, run_id, event)
        await queue.put(event)

    async def producer() -> None:
        try:
            await run_question(
                question=ask_request.question,
                settings=settings,
                history=ask_request.history,
                on_event=on_event,
            )
        except Exception:
            logger.exception('run failed for /ask')
            failed = RunFailed(message='The run failed before it could answer.')
            await log_event(settings, run_id, failed)
            await queue.put(failed)
        finally:
            await queue.put(None)

    async def consumer() -> AsyncGenerator[dict[str, str]]:
        task = asyncio.create_task(producer())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield {'event': event.type, 'data': event.model_dump_json()}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return EventSourceResponse(consumer())


app.mount('/', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='static')
