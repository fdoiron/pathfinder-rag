import sqlite3

import pandas as pd


def build_fts5_index(chunks_df: pd.DataFrame, con: sqlite3.Connection, fts5_tokenchar: bool) -> None:

    tokenize = "unicode61 tokenchars '-'" if fts5_tokenchar else 'unicode61'
    columns = f'chunk_id UNINDEXED, text, category UNINDEXED, tokenize="{tokenize}"'

    con.execute('DROP TABLE IF EXISTS chunks_fts')
    con.execute(f'CREATE VIRTUAL TABLE chunks_fts USING fts5({columns})')

    chunks_df[['chunk_id', 'text', 'category']].to_sql('chunks_fts', con, if_exists='append', index=False)
    con.commit()
