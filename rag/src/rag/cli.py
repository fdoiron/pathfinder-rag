import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rag.chunking import load_tokenizer
from rag.config import get_settings
from rag.corpus import chunk_corpus, embed_corpus
from rag.embedding import LocalEmbedder
from rag.evaluation import collapse_to_urls, evaluate_query, load_queries, write_run
from rag.models import ChunksManifest
from rag.parsing import parse_corpus_dir
from rag.retrieval import ManifestMismatchError, load_retriever

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)


@app.callback()
def _callback() -> None:
    """Pathfinder 1e RAG pipeline CLI."""


@app.command()
def build_corpus(
    html_dir: Annotated[
        Path,
        typer.Argument(
            help='Path to the directory of scraped HTML files',
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(help='Path to the output parquet file', writable=True),
    ] = Path('data/corpus.parquet'),
) -> None:
    """
    Build a corpus from a directory of scraped HTML files and save it as a parquet file.
    """
    settings = get_settings()
    logging.info(f'Parsing HTML files from {html_dir}')
    articles = parse_corpus_dir(html_dir, min_body_length=settings.min_body_length)
    if not articles:
        logging.warning(f'No articles parsed from {html_dir}. Writing empty corpus and chunks')

    logging.info(f'Loading tokenizer {settings.tokenizer_model}')
    tokenizer = load_tokenizer(settings.tokenizer_model)
    logging.info('Chunking articles')
    chunks = chunk_corpus(articles, tokenizer, settings.chunk_max_tokens, settings.chunk_overlap)
    if not chunks:
        logging.warning('No chunks produced. Writing empty chunks file')

    chunks_file = output_file.with_name('chunks.parquet')
    manifest_path = chunks_file.with_suffix('.manifest.json')

    docs_df = pd.DataFrame([a.model_dump(mode='json') for a in articles])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    docs_df.to_parquet(output_file, index=False)
    typer.echo(f'wrote {len(articles)} articles to {output_file}')

    chunks_df = pd.DataFrame([c.model_dump() for c in chunks])

    logging.info(f'Loading embedder {settings.embedding_model}')
    embedder = LocalEmbedder(settings)
    if chunks:
        logging.info(f'Embedding {len(chunks)} chunks')
        chunks_df = embed_corpus(chunks_df, embedder, text_columns=['text'])

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_parquet(chunks_file, index=False)
    typer.echo(f'wrote {len(chunks)} chunks to {chunks_file}')

    manifest = ChunksManifest.build(
        settings, output_file, len(articles), len(chunks), embedder.torch_dtype, embedder.query_prompt
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
    typer.echo(f'wrote manifest to {manifest_path}')


@app.command()
def search(
    query: Annotated[str, typer.Argument(help='Specify search query')],
    k: Annotated[
        int,
        typer.Option(
            help='Maximum number of search results to return',
        ),
    ] = 5,
    embedding_file_path: Annotated[
        Path | None,
        typer.Option(
            help='Path to the embedding parquet file',
            exists=True,
            readable=True,
        ),
    ] = None,
    category: Annotated[str | None, typer.Option(help='restrict to one category, ex: bestiary')] = None,
) -> None:
    """Embeds search query, returns top k results"""
    settings = get_settings()
    embedding_file_path = embedding_file_path if embedding_file_path else settings.chunks_path
    embedder = LocalEmbedder(settings)

    try:
        retriever = load_retriever(
            chunks_file=embedding_file_path,
            embedder=embedder,
            settings=settings,
        )
    except (FileNotFoundError, ManifestMismatchError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    chunk_hits = retriever.search(query=query, k=k, category=category)

    if not chunk_hits:
        typer.echo('No results found.')
        return

    for i, result in enumerate(chunk_hits):
        typer.echo(f'--- Result {i + 1} ---')
        typer.echo(f'Score: {result.score:.3f}')
        typer.echo(f'Title: {result.title}')
        typer.echo(f'URL: {result.url}')


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
        typer.Option(
            help='Path to the embedding parquet file',
            exists=True,
            readable=True,
        ),
    ] = None,
    k: Annotated[
        int,
        typer.Option(
            help='Maximum number of search results to return',
        ),
    ] = 5,
    run_dir: Annotated[Path, typer.Option(help='Directory to save evaluation run results')] = Path('eval/runs'),
) -> None:
    """
    Evaluate the retrieval performance of the corpus.
    """

    try:
        queries = load_queries(queries_file)
    except ValueError as e:
        typer.echo(f'Error loading queries: {e}', err=True)
        raise typer.Exit(1) from e

    settings = get_settings()
    embedding_file_path = embedding_file_path if embedding_file_path else settings.chunks_path
    embedder = LocalEmbedder(settings)

    try:
        retriever = load_retriever(
            chunks_file=embedding_file_path,
            embedder=embedder,
            settings=settings,
        )
    except (FileNotFoundError, ManifestMismatchError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    results = [evaluate_query(query, collapse_to_urls(retriever.search(query.query, k=k * 5), k)) for query in queries]

    run_path, run = write_run(run_dir, retriever.manifest, k, results)
    typer.echo(run.summary.format_line())

    typer.echo('\nby type:')
    for name, group_summary in run.by_type.items():
        typer.echo(f'  {name}: {group_summary.format_line()}')

    typer.echo('\nby category:')
    for name, group_summary in run.by_category.items():
        typer.echo(f'  {name}: {group_summary.format_line()}')

    misses = [r for r in results if r.is_miss]
    if misses:
        typer.echo(f'\n{len(misses)} queries had no hits:')
        for r in misses:
            got = f'{r.retrieved_items[0].url} ({r.retrieved_items[0].score:.2f})' if r.retrieved_items else 'nothing'
            typer.echo(f'  query: {r.query}')
            typer.echo(f'  expected: {r.expected_urls}')
            typer.echo(f'  got: {got}')

    typer.echo(f'\nWrote evaluation run results to {run_path}')


if __name__ == '__main__':
    app()
