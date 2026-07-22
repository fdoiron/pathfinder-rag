import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, HttpUrl

from rag.config import Settings

TaskType = Literal['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY']


class Article(BaseModel):
    model_config = ConfigDict(extra='forbid')
    doc_id: str  # filename slug without .html
    url: HttpUrl  # reconstructed from slug __ -> /
    title: str  # from <h1>
    category: str  # first slug segment
    breadcrumb: list[str]  # from div.breadcrumbs
    body_md: str  # cleaned markdown, canonical text
    n_chars: int  # for filtering/chunking/statistics


class Chunk(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_id: str  # {doc_id}#{i:03d}
    doc_id: str  # foreign key to documents
    heading_path: list[str]  # e.g. ['Aboleth','SPECIAL ABILITIES]
    text: str  # embedded text = title + heading path + body
    category: str  # copied from the document for filtering
    n_tokens: int  # measured by embedding's tokenizer


class SearchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    article: Article
    score: float


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_file: str
    source_sha256: str
    n_articles: int
    text_columns: list[str]
    embedding_model: str
    embedding_dim: int
    task_type: TaskType
    created_at: datetime

    @classmethod
    def build(
        cls, settings: Settings, df: pd.DataFrame, source_file: str | Path, text_columns: list[str]
    ) -> 'CorpusManifest':
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


class ChunksManifest(BaseModel):
    """Provenance for chunks.parquet: parameters and source parsed data"""

    model_config = ConfigDict(extra='forbid')
    source_file: str  # the documents parquet the chunks were cut from
    source_sha256: str
    n_articles: int
    n_chunks: int
    min_body_length: int
    tokenizer_model: str
    max_tokens: int
    overlap: int
    parser_version: str  # from parsing.PARSER_VERSION
    embedding_model: str
    embedding_dim: int
    embedding_dtype: str  # compute dtype the weights were loaded in (device-dependent, affects vectors)
    query_prompt: str
    created_at: datetime

    @classmethod
    def build(
        cls,
        settings: Settings,
        source_file: str | Path,
        n_articles: int,
        n_chunks: int,
        embedding_dtype: str,
        query_prompt: str,
    ) -> 'ChunksManifest':
        from rag.parsing import PARSER_VERSION  # local import: parsing imports this module

        return cls(
            source_file=str(source_file),
            source_sha256=hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
            n_articles=n_articles,
            n_chunks=n_chunks,
            min_body_length=settings.min_body_length,
            tokenizer_model=settings.tokenizer_model,
            max_tokens=settings.chunk_max_tokens,
            overlap=settings.chunk_overlap,
            parser_version=PARSER_VERSION,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            embedding_dtype=embedding_dtype,
            query_prompt=query_prompt,
            created_at=datetime.now(UTC),
        )


class Embedder(Protocol):
    def embed(self, texts: list[str], task_type: TaskType = ...) -> np.ndarray: ...
