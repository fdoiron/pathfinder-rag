"""Sweeps BM25 title/text weight ratios against the eval query set

Writes its own summary file since ChunksManifest is corpus build provenance only and has no field for query time
weights
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from rag.config import get_settings
from rag.embedding import load_embedder
from rag.evaluation import EvalSummary, evaluate_query, load_queries, search_top_k_docs, summarize_results
from rag.retrieval import SearchMethod, load_retriever

QUERIES_PATH = Path('eval/queries.jsonl')
RUN_DIR = Path('eval/runs')
K = 5
METHOD: SearchMethod = 'hybrid'

WEIGHT_GRID: list[tuple[float, float]] = [
    (10.0, 1.0),  # current default
    (7.0, 1.0),
    (5.0, 1.0),
    (3.0, 1.0),
    (2.0, 1.0),
    (1.0, 1.0),
]


def main() -> None:
    queries = load_queries(QUERIES_PATH)
    base_settings = get_settings()
    embedder = load_embedder(base_settings)

    rows: list[tuple[float, float, EvalSummary]] = []
    for title_weight, text_weight in WEIGHT_GRID:
        settings = base_settings.model_copy(update={'fts5_title_weight': title_weight, 'fts5_text_weight': text_weight})
        retriever = load_retriever(settings.chunks_path, embedder, settings)
        results = [evaluate_query(q, search_top_k_docs(retriever, q.query, K, method=METHOD)) for q in queries]
        summary = summarize_results(results)
        rows.append((title_weight, text_weight, summary))
        print(f'title={title_weight:5.1f}  text={text_weight:5.1f}  {summary.format_line()}')

    best_title, best_text, best_summary = max(rows, key=lambda row: row[2].mrr)
    print(f'\nbest by MRR: title={best_title}  text={best_text}  {best_summary.format_line()}')

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f'{datetime.now(UTC):%Y-%m-%dT%H-%M-%S}_fts5_weight_sweep.json'
    payload = {
        'created_at': datetime.now(UTC).isoformat(),
        'queries_file': str(QUERIES_PATH),
        'k': K,
        'method': METHOD,
        'runs': [
            {'title_weight': title_weight, 'text_weight': text_weight, 'summary': summary.model_dump()}
            for title_weight, text_weight, summary in rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\nwrote sweep results to {out_path}')


if __name__ == '__main__':
    main()
