import numpy as np

from rag.models import Embedder


def search_vector(
    matrix: np.ndarray,
    chunk_ids: np.ndarray,
    categories: np.ndarray,
    embedder: Embedder,
    query: str,
    k: int,
    category: str | None = None,
) -> list[tuple[str, float]]:
    """Returns [(chunk_id, cosine_score), ...] best-first.

    matrix rows must be L2-normalized and row-aligned with chunk_ids/categories.
    """
    q = embedder.embed([query], task_type='RETRIEVAL_QUERY')[0]
    if not np.isfinite(q).all():
        raise ValueError(f'embedder returned a non finite (NaN/inf) vector for query {query!r}')
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        raise ValueError(f'embedder returned a zero-norm vector for query {query!r}')
    q = q / q_norm  # normalize to length 1.0
    scores = matrix @ q  # dot product of two normalized vectors = cosine similarity
    if category is not None:
        scores = np.where(categories == category, scores, -np.inf)
    top_results = [i for i in np.argsort(scores)[::-1][:k] if scores[i] > -np.inf]
    return [(chunk_ids[i], float(scores[i])) for i in top_results]
