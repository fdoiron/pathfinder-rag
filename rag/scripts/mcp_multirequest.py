import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = 'http://127.0.0.1:8000/mcp'

QUERIES = [
    {'query': 'red dragon', 'k': 5},
    {'query': 'fireball spell', 'k': 5},
    {'query': 'pack tactics', 'k': 5},
    {'query': 'paladin', 'k': 5},
    {'query': 'sorcerer', 'k': 5},
    {'query': 'rogue', 'k': 5},
    {'query': 'barbarian', 'k': 5},
    {'query': 'magus', 'k': 5},
    {'query': 'druid', 'k': 5},
]


async def fire_one(session: ClientSession, args: dict) -> None:
    _ = await session.call_tool('rag_search', arguments={'search_query': args})
    print(f'done: {args["query"]!r}')


async def main() -> None:
    async with streamable_http_client(SERVER_URL) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        await asyncio.gather(*(fire_one(session, q) for q in QUERIES))


if __name__ == '__main__':
    asyncio.run(main())
