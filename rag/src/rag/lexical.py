import sqlite3

import pandas as pd


def build_fts5_index(chunks_df: pd.DataFrame, con: sqlite3.Connection, fts5_tokenchar: bool) -> None:

    tokenize = "unicode61 tokenchars '-'" if fts5_tokenchar else 'unicode61'
    columns = f'chunk_id UNINDEXED, title, text, category UNINDEXED, tokenize="{tokenize}"'

    con.execute('DROP TABLE IF EXISTS chunks_fts')
    con.execute(f'CREATE VIRTUAL TABLE chunks_fts USING fts5({columns})')

    # Heading path -> which page did the chunk originate from (ex: 'Power Attack (Combat)')
    # Seperation lets bm25 score title and body with different weights
    fts_df = chunks_df[['chunk_id', 'text', 'category']].copy()
    fts_df.insert(1, 'title', chunks_df['heading_path'].str[0].fillna(''))

    fts_df[['chunk_id', 'title', 'text', 'category']].to_sql('chunks_fts', con, if_exists='append', index=False)
    con.commit()


def search_fts5(
    con: sqlite3.Connection,
    query: str,
    k: int,
    title_weight: float,
    text_weight: float,
) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_score), ...] best-first. Scores are negative; more negative = better."""
    # bm25() needs one weight per column in order even for UNINDEXED
    # Skip a slot -> the rest of the weights shift onto the wrong columns
    return con.execute(
        'SELECT chunk_id, bm25(chunks_fts, 1.0, ?, ?, 1.0) AS score FROM chunks_fts '
        'WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?',
        [title_weight, text_weight, query, k],
    ).fetchall()
