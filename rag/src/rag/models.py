import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, HttpUrl

from rag.config import Settings


class Article(BaseModel):
    title: str
    url: HttpUrl
    body: str


class SearchResult(BaseModel):
    article: Article
    score: float


class CorpusManifest(BaseModel):
    source_file: str
    source_sha256: str
    n_articles: int
    text_columns: list[str]
    embedding_model: str
    embedding_dim: int
    task_type: str
    created_at: datetime

    @classmethod
    def build(cls, settings: Settings, df: pd.DataFrame, source_file, text_columns: list[str]) -> 'CorpusManifest':
        return cls(
            source_file=str(source_file),
            source_sha256=hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
            n_articles=len(df),
            text_columns=text_columns,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            task_type='RETRIEVAL_DOCUMENT',
            created_at=datetime.now(UTC),
        )


class Embedder(Protocol):
    def embed(self, texts: list[str], task_type: str = ...) -> np.ndarray: ...
