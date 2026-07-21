import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rag.chunking import load_tokenizer
from rag.config import get_settings
from rag.corpus import chunk_corpus, embed_corpus
from rag.embedding import LocalEmbedder
from rag.models import ChunksManifest
from rag.parsing import parse_corpus_dir

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

    manifest = ChunksManifest.build(settings, output_file, len(articles), len(chunks), embedder.query_prompt)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
    typer.echo(f'wrote manifest to {manifest_path}')


if __name__ == '__main__':
    app()
