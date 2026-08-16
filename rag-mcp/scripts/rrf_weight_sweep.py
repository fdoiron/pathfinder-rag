"""Sweeps RRF fusion weight (how much hybrid trusts vector vs. BM25) against the eval query set"""

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

# (vector_weight, bm25_weight)
WEIGHT_GRID: list[tuple[float, float]] = [
    (1.0, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
    (5.0, 1.0),
    (10.0, 1.0),
    (15.0, 1.0),
    (20.0, 1.0),
    (30.0, 1.0),
]


def main() -> None:
    queries = load_queries(QUERIES_PATH)
    base_settings = get_settings()
    embedder = load_embedder(base_settings)

    rows: list[tuple[float, float, EvalSummary]] = []
    for vector_weight, bm25_weight in WEIGHT_GRID:
        settings = base_settings.model_copy(update={'rrf_vector_weight': vector_weight, 'rrf_bm25_weight': bm25_weight})
        retriever = load_retriever(settings.chunks_path, embedder, settings)
        results = [evaluate_query(q, search_top_k_docs(retriever, q.query, K, method=METHOD)) for q in queries]
        summary = summarize_results(results)
        rows.append((vector_weight, bm25_weight, summary))
        print(f'vector={vector_weight:5.1f}  bm25={bm25_weight:5.1f}  {summary.format_line()}')

    best_vector, best_bm25, best_summary = max(rows, key=lambda row: row[2].mrr)
    print(f'\nbest by MRR: vector={best_vector}  bm25={best_bm25}  {best_summary.format_line()}')

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f'{datetime.now(UTC):%Y-%m-%dT%H-%M-%S}_rrf_weight_sweep.json'
    payload = {
        'created_at': datetime.now(UTC).isoformat(),
        'queries_file': str(QUERIES_PATH),
        'k': K,
        'method': METHOD,
        'runs': [
            {'vector_weight': vector_weight, 'bm25_weight': bm25_weight, 'summary': summary.model_dump()}
            for vector_weight, bm25_weight, summary in rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\nwrote sweep results to {out_path}')


if __name__ == '__main__':
    main()
