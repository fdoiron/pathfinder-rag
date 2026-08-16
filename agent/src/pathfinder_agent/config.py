from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_', env_file='.env', env_file_encoding='utf-8')

    mcp_server_url: str = 'http://localhost:8000/mcp'  # the MCP server to dial
    mcp_auth_token: Annotated[SecretStr, Field(min_length=1)] | None = None  # None means the server has auth disabled
    llm_base_url: str = 'http://localhost:11434/v1'  # OpenAI-compatible endpoint for Ollama
    llm_model: str = 'qwen3:14b'

    agent_max_iters: PositiveInt = 8  # tool calling hops before giving up regardless of wall clock
    agent_hop_timeout: PositiveFloat = 30.0  # seconds allowed for one LLM call or one tool call, also the SDK timeout
    agent_wall_clock_timeout: PositiveFloat = 120.0  # seconds for the whole loop. Overrides max_iters
    agent_context_token_budget: PositiveInt = 4000  # estimated tool-result tokens before the loop stops calling tools
    agent_max_tool_attempts: PositiveInt = 2  # tool attempts (1 initial + retries) for retryable error/hop timeout
    agent_retry_backoff_base: PositiveFloat = 1.0  # seconds, doubles by retry
    agent_system_prompt_path: Path | None = None  # override for the packaged prompts/agent_system.txt for tests

    # None -> OTLPSpanExporter falls back to its own env lookup, defaulting to http://localhost:4317
    otel_exporter_otlp_endpoint: Annotated[str | None, Field(validation_alias='OTEL_EXPORTER_OTLP_ENDPOINT')] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
