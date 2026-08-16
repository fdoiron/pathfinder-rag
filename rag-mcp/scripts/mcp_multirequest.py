import asyncio
from collections import Counter

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from rag.config import get_settings

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
    {'query': 'red dragon', 'k': 5},
    {'query': 'fireball spell', 'k': 5},
    {'query': 'pack tactics', 'k': 5},
    {'query': 'paladin', 'k': 5},
    {'query': 'sorcerer', 'k': 5},
    {'query': 'rogue', 'k': 5},
    {'query': 'barbarian', 'k': 5},
    {'query': 'magus', 'k': 5},
    {'query': 'druid', 'k': 5},
    {'query': 'red dragon', 'k': 5},
    {'query': 'fireball spell', 'k': 5},
    {'query': 'pack tactics', 'k': 5},
    {'query': 'paladin', 'k': 5},
    {'query': 'sorcerer', 'k': 5},
    {'query': 'rogue', 'k': 5},
    {'query': 'barbarian', 'k': 5},
    {'query': 'magus', 'k': 5},
    {'query': 'druid', 'k': 5},
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


async def fire_one(session: ClientSession, args: dict) -> str:
    """Outcome of one call: 'ok', the SearchResults.error_category the server sent back, or 'unclassified'
    for a genuine protocol-level failure (bad arguments) that never reached rag_search's body."""
    result = await session.call_tool('rag_search', arguments={'search_query': args})
    if result.is_error:
        text = result.content[0].text if result.content else ''
        print(f'unclassified  {args["query"]!r}: {text}')
        return 'unclassified'

    category = (result.structured_content or {}).get('error_category')
    if category is None:
        return 'ok'
    message = (result.structured_content or {}).get('message')
    print(f'{category:<13} {args["query"]!r}: {message}')
    return category


async def main() -> None:
    token = get_settings().mcp_auth_token
    headers = {'Authorization': f'Bearer {token.get_secret_value()}'} if token else {}
    async with (
        create_mcp_http_client(headers) as http_client,
        streamable_http_client(SERVER_URL, http_client=http_client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        outcomes = Counter(await asyncio.gather(*(fire_one(session, q) for q in QUERIES)))

    summary = ', '.join(f'{count} {name}' for name, count in outcomes.most_common())
    print(f'\n{sum(outcomes.values())} calls -> {summary}')


if __name__ == '__main__':
    asyncio.run(main())
