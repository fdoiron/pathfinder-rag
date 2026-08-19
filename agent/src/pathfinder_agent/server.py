import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pathfinder_agent.config import Settings, get_settings
from pathfinder_agent.models import AgentEvent, QueueItem, RunFailed
from pathfinder_agent.runner import run_question

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx2').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=1000)]


@app.post('/ask')
async def ask(ask_request: AskRequest, settings: Annotated[Settings, Depends(get_settings)]) -> EventSourceResponse:
    queue: asyncio.Queue[QueueItem | None] = asyncio.Queue()

    async def on_event(event: AgentEvent) -> None:
        await queue.put(event)

    async def producer() -> None:
        try:
            await run_question(question=ask_request.question, settings=settings, on_event=on_event)
        except Exception:
            logger.exception('run failed for /ask')
            await queue.put(RunFailed(message='The run failed before it could answer.'))
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
