import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import typer

from rag.answer import LLMUnavailableError, answer_question, make_llm_client
from rag.config import Settings, get_settings
from rag.evaluation import evaluate_query, load_queries, search_top_k_docs, write_run
from rag.lexical import build_fts5_index
from rag.models import ChunksManifest
from rag.parsing import parse_corpus_dir
from rag.retrieval import ManifestMismatchError, OrphanChunksError, SearchMethod, StaleIndexError, load_retriever

if TYPE_CHECKING:
    from rag.embedding import LocalEmbedder

# torch/transformers imported inside each command body
app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)


@app.callback()
def _callback() -> None:
    """Pathfinder 1e RAG pipeline CLI."""


def _require_positive(value: float | None) -> float | None:
    # mirrors config.py's PositiveFloat; model_copy(update=...) in _apply_fts5_weight_overrides skips validation
    if value is not None and value <= 0:
        raise typer.BadParameter('must be > 0')
    return value


def _load_embedder(settings: Settings) -> 'LocalEmbedder':
    from rag.embedding import EmbedderUnavailableError, load_embedder

    try:
        return load_embedder(settings)
    except EmbedderUnavailableError as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e


def _apply_fts5_weight_overrides(
    settings: Settings, fts5_title_weight: float | None, fts5_text_weight: float | None
) -> Settings:
    overrides = {
        name: value
        for name, value in [('fts5_title_weight', fts5_title_weight), ('fts5_text_weight', fts5_text_weight)]
        if value is not None
    }
    return settings.model_copy(update=overrides) if overrides else settings


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
) -> None:
    """
    Build a corpus from a directory of scraped HTML files and save it as a parquet file.

    Writes the documents parquet, the chunks parquet and its manifest to the paths configured in
    Settings (RAG_CORPUS_PATH / RAG_CHUNKS_PATH)
    """
    from rag.chunking import load_tokenizer
    from rag.corpus import chunk_corpus, embed_corpus

    settings = get_settings()
    logging.info(f'Parsing HTML files from {html_dir}')
    articles = parse_corpus_dir(html_dir, min_body_length=settings.min_body_length)
    if not articles:
        typer.echo(f'Error: no articles parsed from {html_dir}', err=True)
        raise typer.Exit(code=1)

    logging.info(f'Loading tokenizer {settings.tokenizer_model}')
    tokenizer = load_tokenizer(settings.tokenizer_model)
    logging.info('Chunking articles')
    chunks = chunk_corpus(articles, tokenizer, settings.chunk_max_tokens, settings.chunk_overlap)
    if not chunks:
        logging.warning('No chunks produced. Writing empty chunks file')

    corpus_file = settings.corpus_path
    chunks_file = settings.chunks_path
    manifest_path = chunks_file.with_suffix('.manifest.json')

    docs_df = pd.DataFrame([a.model_dump(mode='json') for a in articles])
    corpus_file.parent.mkdir(parents=True, exist_ok=True)
    docs_df.to_parquet(corpus_file, index=False)
    typer.echo(f'wrote {len(articles)} articles to {corpus_file}')

    chunks_df = pd.DataFrame([c.model_dump() for c in chunks])

    logging.info(f'Loading embedder {settings.embedding_model}')
    embedder = _load_embedder(settings)
    if chunks:
        logging.info(f'Embedding {len(chunks)} chunks')
        chunks_df = embed_corpus(chunks_df, embedder, text_columns=['text'])

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_parquet(chunks_file, index=False)
    typer.echo(f'wrote {len(chunks)} chunks to {chunks_file}')

    fts_path = chunks_file.with_suffix('.fts5.db')
    fts_con = sqlite3.connect(fts_path)
    build_fts5_index(chunks_df, fts_con, fts5_tokenchar=settings.fts5_tokenchar)
    fts_con.close()
    typer.echo(f'wrote fts5 index to {fts_path}')

    manifest = ChunksManifest.build(
        settings, corpus_file, len(articles), len(chunks), embedder.torch_dtype, embedder.query_prompt
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
            min=1,
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
    method: Annotated[SearchMethod, typer.Option(help='retrieval method')] = 'hybrid',
    fts5_title_weight: Annotated[
        float | None,
        typer.Option(
            callback=_require_positive,
            help='bm25() weight for chunk headings vs. body text (defaults to settings.fts5_title_weight)',
        ),
    ] = None,
    fts5_text_weight: Annotated[
        float | None,
        typer.Option(
            callback=_require_positive,
            help='bm25() weight for chunk body text (defaults to settings.fts5_text_weight)',
        ),
    ] = None,
) -> None:
    """Embeds search query, returns top k results"""
    settings = get_settings()
    settings = _apply_fts5_weight_overrides(settings, fts5_title_weight, fts5_text_weight)
    embedding_file_path = embedding_file_path if embedding_file_path else settings.chunks_path
    embedder = _load_embedder(settings)

    try:
        retriever = load_retriever(
            chunks_file=embedding_file_path,
            embedder=embedder,
            settings=settings,
        )
    except (FileNotFoundError, ManifestMismatchError, OrphanChunksError, StaleIndexError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    chunk_hits = retriever.search(query=query, k=k, category=category, method=method)

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
            min=1,
        ),
    ] = 50,
    run_dir: Annotated[Path, typer.Option(help='Directory to save evaluation run results')] = Path('eval/runs'),
    method: Annotated[SearchMethod, typer.Option(help='retrieval method')] = 'hybrid',
    fts5_title_weight: Annotated[
        float | None,
        typer.Option(
            callback=_require_positive,
            help='bm25() weight for chunk headings vs. body text (defaults to settings.fts5_title_weight)',
        ),
    ] = None,
    fts5_text_weight: Annotated[
        float | None,
        typer.Option(
            callback=_require_positive,
            help='bm25() weight for chunk body text (defaults to settings.fts5_text_weight)',
        ),
    ] = None,
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
    settings = _apply_fts5_weight_overrides(settings, fts5_title_weight, fts5_text_weight)
    embedding_file_path = embedding_file_path if embedding_file_path else settings.chunks_path
    embedder = _load_embedder(settings)

    try:
        retriever = load_retriever(
            chunks_file=embedding_file_path,
            embedder=embedder,
            settings=settings,
        )
    except (FileNotFoundError, ManifestMismatchError, OrphanChunksError, StaleIndexError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    results = [evaluate_query(query, search_top_k_docs(retriever, query.query, k, method=method)) for query in queries]

    run_path, run = write_run(run_dir, retriever.manifest, method, k, results)
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


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help='a rules question')],
    k: Annotated[
        int | None,
        typer.Option(
            help='Number of excerpts to retrieve for the prompt (defaults to settings.ask_k)',
            min=1,
        ),
    ] = None,
    embedding_file_path: Annotated[
        Path | None,
        typer.Option(
            help='Path to the embedding parquet file',
            exists=True,
            readable=True,
        ),
    ] = None,
    category: Annotated[str | None, typer.Option(help='restrict to one category, ex: bestiary')] = None,
    method: Annotated[SearchMethod, typer.Option(help='retrieval method')] = 'hybrid',
) -> None:
    """Answer a rules question with numbered d20pfsrd citations."""
    settings = get_settings()
    embedding_file_path = embedding_file_path if embedding_file_path else settings.chunks_path
    embedder = _load_embedder(settings)

    try:
        retriever = load_retriever(embedding_file_path, embedder, settings)
    except (FileNotFoundError, ManifestMismatchError, OrphanChunksError, StaleIndexError) as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    try:
        result = answer_question(
            question, retriever, make_llm_client(settings), settings, k=k, category=category, method=method
        )
    except LLMUnavailableError as e:
        typer.echo(f'Error: {e}', err=True)
        raise typer.Exit(code=1) from e

    typer.echo(result.text)
    typer.echo()
    for c in result.citations:
        typer.echo(f'[{c.n}] {c.title} — {" > ".join(c.heading_path)}')
        typer.echo(f'    {c.url}')


if __name__ == '__main__':
    app()
