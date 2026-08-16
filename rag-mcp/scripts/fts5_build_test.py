import sqlite3
import time
from pathlib import Path

import pandas as pd

from rag.config import get_settings
from rag.evaluation import load_queries
from rag.lexical import build_fts5_index, search_fts5

QUERIES_PATH = Path('eval/queries.jsonl')


# load exact_name queries from eval queries.jsonl and provides top 5 results for bm25
def main() -> None:
    settings = get_settings()

    chunks_df = pd.read_parquet(settings.chunks_path, columns=['chunk_id', 'text', 'category', 'heading_path'])
    print(f'{len(chunks_df)} chunks')

    fts_path = settings.chunks_path.with_suffix('.fts5.db')
    con = sqlite3.connect(fts_path)

    start = time.perf_counter()
    build_fts5_index(chunks_df, con, fts5_tokenchar=settings.fts5_tokenchar)
    print(f'built {fts_path} in {time.perf_counter() - start:.1f}s')

    row_count = con.execute('SELECT COUNT(*) FROM chunks_fts').fetchone()[0]
    print(f'indexed {row_count} rows (expected {len(chunks_df)})')
    assert row_count == len(chunks_df), 'row count mismatch'

    exact_name_queries = [q for q in load_queries(QUERIES_PATH) if q.type == 'exact_name']
    for eval_query in exact_name_queries:
        print(f'\nquery: {eval_query.query!r}  (expected {eval_query.expected_urls})')
        rows = search_fts5(
            con,
            eval_query.query,
            k=5,
            title_weight=settings.fts5_title_weight,
            text_weight=settings.fts5_text_weight,
        )
        for chunk_id, score in rows:
            print(f'  {score:7.3f}  {chunk_id}')

    con.close()


if __name__ == '__main__':
    main()
