import pandas as pd

from rag.models import Embedder


def embed_corpus(df: pd.DataFrame, embedder: Embedder, text_columns: list[str]) -> pd.DataFrame:
    """embeds concatenated text_columns into a dataframe"""
    texts = df[text_columns].agg('\n'.join, axis=1).tolist()
    vectors = embedder.embed(texts, task_type='RETRIEVAL_DOCUMENT')
    out = df.copy()
    out['embedding'] = list(vectors)
    return out
