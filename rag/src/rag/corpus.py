import logging

import pandas as pd
from transformers import PreTrainedTokenizerBase

from rag.chunking import chunk_article
from rag.models import Article, Chunk, Embedder

logger = logging.getLogger(__name__)


def chunk_corpus(
    articles: list[Article], tokenizer: PreTrainedTokenizerBase, max_tokens: int, overlap: int
) -> list[Chunk]:

    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, tokenizer, max_tokens=max_tokens, overlap=overlap))

    hard_limit = int(max_tokens * 1.02)
    broken = [c.chunk_id for c in chunks if c.n_tokens > hard_limit]
    if broken:
        raise ValueError(f'{len(broken)} chunks exceed max_tokens={max_tokens} beyond BPE slack: {broken[:5]}')

    drifted = [c.n_tokens for c in chunks if c.n_tokens > max_tokens]
    if drifted:
        logger.warning(
            f'{len(drifted)}/{len(chunks)} chunks over max_tokens={max_tokens} within BPE slack, worst={max(drifted)}'
        )

    return chunks


def embed_corpus(df: pd.DataFrame, embedder: Embedder, text_columns: list[str]) -> pd.DataFrame:
    """embeds concatenated text_columns into a dataframe"""
    texts = df[text_columns].agg('\n'.join, axis=1).tolist()
    vectors = embedder.embed(texts, task_type='RETRIEVAL_DOCUMENT')
    out = df.copy()
    out['embedding'] = list(vectors)
    return out
