from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_', env_file='.env', env_file_encoding='utf-8')

    gcp_service_account_file: Path | None = Path('gcp-service-account-file.json')
    gcp_project: str | None = None
    gcp_location: str = 'northamerica-northeast1'
    embedding_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    corpus_path: Path = Path('data/corpus.parquet')
    embedded_corpus_path: Path = Path('data/corpus_embedded.parquet')
    min_body_length: int = 100
    tokenizer_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    chunk_max_tokens: int = 450
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
