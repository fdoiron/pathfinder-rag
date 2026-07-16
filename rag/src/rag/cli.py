import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rag.config import get_settings
from rag.parsing import parse_corpus_dir

app = typer.Typer()
logging.basicConfig(level=logging.INFO)


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
):
    """
    Build a corpus from a directory of scraped HTML files and save it as a parquet file.
    """
    settings = get_settings()
    logging.info(f'Parsing HTML files from {html_dir}')
    articles = parse_corpus_dir(html_dir, min_body_length=settings.min_body_length)
    df = pd.DataFrame([a.model_dump(mode='json') for a in articles])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    typer.echo(f'wrote {len(articles)} articles to {output_file}')


if __name__ == '__main__':
    app()
