import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rag.config import get_settings
from rag.corpus import embed_corpus
from rag.embedding import VertexEmbedder
from rag.evaluation import evaluate_query, load_queries, summarize_results, write_run
from rag.genai_client import make_genai_client
from rag.models import CorpusManifest

# from rag.parsing import parse_corpus
from rag.retrieval import ManifestMismatchError, Retriever, load_retriever

app = typer.Typer()
logging.basicConfig(level=logging.INFO)


def _build_retriever(embedding_file: Path | None) -> Retriever:
    settings = get_settings()
    embedding_file = embedding_file or settings.embedded_corpus_path
    client = make_genai_client(settings)
    embedder = VertexEmbedder(client, settings)
    try:
        return load_retriever(embedding_file, embedder, settings)
    except (FileNotFoundError, ManifestMismatchError) as e:
        typer.echo(f'Error loading retriever: {e}', err=True)
        raise typer.Exit(1) from e


# @app.command()
# def build_corpus(
#     input_file: Annotated[
#         Path,
#         typer.Argument(
#             help='Path to the input markdown file',  # TODO update once HTML from scraper is done
#             exists=True,
#             readable=True,
#         ),
#     ],
#     output_file: Annotated[
#         Path,
#         typer.Option(help='Path to the output parquet file', writable=True),
#     ] = Path('data/corpus.parquet'),
# ):
#     """
#     Build a corpus from a markdown file and save it as a parquet file.
#     """
#     settings = get_settings()
#     logging.info(f'Reading markdown file from {input_file}')
#     articles = parse_corpus(input_file.read_text(encoding='utf-8'), min_body_length=settings.min_body_length)
#     df = pd.DataFrame([a.model_dump(mode='json') for a in articles])
#     output_file.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(output_file, index=False)
#     typer.echo(f'wrote {len(articles)} articles to {output_file}')


@app.command()
def embed(
    input_file: Annotated[
        Path | None, typer.Argument(help='Path to the input parquet file', exists=True, readable=True)
    ] = None,
    output_file: Annotated[Path | None, typer.Option(help='Path to the output parquet file', writable=True)] = None,
) -> None:
    """
    Embed a corpus from a parquet file and save it as a parquet file.
    """
    settings = get_settings()
    if input_file is None:
        input_file = settings.corpus_path
    if output_file is None:
        output_file = settings.embedded_corpus_path
    df = pd.read_parquet(input_file)
    client = make_genai_client(settings)
    embedder = VertexEmbedder(client, settings)
    out = embed_corpus(df, embedder, text_columns=['title', 'body'])
    manifest = CorpusManifest.build(settings, out, source_file=input_file, text_columns=['title', 'body'])
    manifest_path = output_file.with_suffix('.manifest.json')
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
    out.to_parquet(output_file, index=False)
    typer.echo(f'wrote {len(out)} articles to {output_file}')
    typer.echo(f'wrote manifest to {manifest_path}')


@app.command()
def search(
    query: Annotated[str, typer.Argument(help='Specify search query')],
    k: Annotated[int, typer.Option(help='Maximum number of results to return')] = 5,
    embedding_file_path: Annotated[
        Path | None,
        typer.Option(help='Path to the embedding parquet file', exists=True, readable=True),
    ] = None,
) -> None:

    retriever = _build_retriever(embedding_file_path)

    results = retriever.search(query, k)
    for i, result in enumerate(results):
        typer.echo(f'--- Result {i + 1} ---')
        typer.echo(f'Score: {result.score:.3f}')
        typer.echo(f'Title: {result.article.title}')
        typer.echo(f'URL: {result.article.url}')


@app.command()
def evaluate(
    queries_file: Annotated[
        Path,
        typer.Argument(
            help='Path to the input queries JSON file for evaluation',
            exists=True,
            readable=True,
        ),
    ],
    embedding_file_path: Annotated[
        Path | None,
        typer.Option(help='Path to the embedding parquet file', exists=True, readable=True),
    ] = None,
    k: Annotated[int, typer.Option(help='Maximum number of results to return')] = 5,
    run_dir: Annotated[Path, typer.Option(help='Directory to save evaluation run results')] = Path('eval/runs'),
) -> None:
    """
    Evaluate the retrieval performance of the corpus.
    """
    retriever = _build_retriever(embedding_file_path)

    try:
        queries = load_queries(queries_file)
    except ValueError as e:
        typer.echo(f'Error loading queries: {e}', err=True)
        raise typer.Exit(1) from e

    results = [evaluate_query(query, retriever.search(query.query, k=k)) for query in queries]
    summary = summarize_results(results)
    typer.echo(summary.format_line())

    misses = [r for r in results if r.is_miss]
    if misses:
        typer.echo(f'\n{len(misses)} queries had no hits:')
        for r in misses:
            got = f'{r.retrieved_items[0].url} ({r.retrieved_items[0].score:.2f})' if r.retrieved_items else 'nothing'
            typer.echo(f'  query: {r.query}')
            typer.echo(f'  expected: {r.expected_urls}')
            typer.echo(f'  got: {got}')

    run_path = write_run(run_dir, retriever.manifest, k, summary, results)
    typer.echo(f'\nWrote evaluation run results to {run_path}')


if __name__ == '__main__':
    app()
