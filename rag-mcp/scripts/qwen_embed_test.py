import numpy as np

from rag.config import get_settings
from rag.embedding import LocalEmbedder

# Reference from the Qwen3-Embedding-0.6B github
EXPECTED = np.array(
    [
        [0.7646, 0.1414],
        [0.1355, 0.6000],
    ]
)

queries = [
    'What is the capital of China?',
    'Explain gravity',
]
documents = [
    'The capital of China is Beijing.',
    (
        'Gravity is a force that attracts two bodies towards each other. '
        'It gives weight to physical objects and is responsible for the '
        'movement of planets around the sun.'
    ),
]


def main() -> None:
    settings = get_settings()
    embedder = LocalEmbedder(settings)

    query_vecs = embedder.embed(queries, task_type='RETRIEVAL_QUERY')
    doc_vecs = embedder.embed(documents, task_type='RETRIEVAL_DOCUMENT')

    scores = query_vecs @ doc_vecs.T

    print('got:')
    print(scores)
    print('\nexpected:')
    print(EXPECTED)

    diff = np.abs(scores - EXPECTED)
    print(f'\nmax abs diff: {diff.max():.4f}')

    if np.allclose(scores, EXPECTED, atol=0.01):
        print('PASS')
    else:
        print('FAIL')


if __name__ == '__main__':
    main()
