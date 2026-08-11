from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_', env_file='.env', env_file_encoding='utf-8')

    embedding_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    reranker_model: str = 'Qwen/Qwen3-Reranker-0.6B'
    embedding_dim: PositiveInt = 1024
    embedding_batch_size: PositiveInt = 32
    reranker_batch_size: PositiveInt = 32
    corpus_path: Path = Path('data/corpus.parquet')
    chunks_path: Path = Path('data/chunks.parquet')
    min_body_length: NonNegativeInt = 100  # 0 keeps every article
    tokenizer_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    chunk_max_tokens: PositiveInt = 450
    chunk_overlap: NonNegativeInt = 50  # 0 disables the carry between windows
    llm_base_url: str = 'http://localhost:11434/v1'  # OpenAI-compatible endpoint for Ollama
    llm_model: str = 'qwen3:14b'
    llm_timeout: PositiveFloat = 60.0  # Ollama timeout in seconds instead of default 10 minutes
    ask_prompt_path: Path | None = None  # override for the packaged prompts/ask.txt, mainly for tests
    ask_k: PositiveInt = 5  # excerpts per prompt
    rerank_fetch_k: PositiveInt = 50  # candidates fused/retrieved and reranked before cutting down to k
    rrf_k: PositiveInt = 60  # Reciprocal Rank Fusion k denominator
    rrf_vector_weight: PositiveFloat = 15.0  # per-list weight on the vector ranking's RRF contribution
    rrf_bm25_weight: PositiveFloat = 1.0  # per-list weight on the BM25 ranking's RRF contribution
    fts5_tokenchar: bool = False  # True: tokenize="unicode61 tokenchars '-'" (keeps "aboleth-psionic" as one token)
    fts5_title_weight: PositiveFloat = 10.0  # bm25() weight for the chunk's heading vs. body text
    fts5_text_weight: PositiveFloat = 1.0
    mcp_drain_timeout: PositiveFloat = 10.0  # seconds to let queued searches finish on shutdown before abandoning them
    mcp_auth_token: Annotated[SecretStr, Field(min_length=1)] | None = None  # None disables auth entirely
    mcp_server_url: str = 'http://localhost:8000'  # this server's own identity in the OAuth resource metadata
    # None -> OTLPSpanExporter falls back to its own env lookup, defaulting to http://localhost:4317
    otel_exporter_otlp_endpoint: Annotated[str | None, Field(validation_alias='OTEL_EXPORTER_OTLP_ENDPOINT')] = None
    otel_console_export: bool = False  # RAG_OTEL_CONSOLE_EXPORT: print metrics via ConsoleMetricExporter


@lru_cache
def get_settings() -> Settings:
    return Settings()
