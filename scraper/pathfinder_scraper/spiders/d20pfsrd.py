import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import pandas as pd
import scrapy
from scrapy.http import Response


class D20pfsrdSpider(scrapy.Spider):
    name = 'd20pfsrd'
    allowed_domains: ClassVar[list[str]] = ['d20pfsrd.com']

    custom_settings = {  # noqa: RUF012
        'JOBDIR': 'crawls/d20pfsrd',  # enables pause/resume
    }

    def __init__(
        self,
        links_path: str = 'd20pfsrd_links.parquet',
        output_dir: str = 'data/html',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.links_path = links_path

    async def start(self) -> AsyncIterator[scrapy.Request]:
        df = pd.read_parquet(self.links_path)
        urls = df['url'].tolist()
        # skip URLs where output files already exists
        new_urls = [url for url in urls if not (self.output_dir / self.url_to_filename(url)).exists()]
        self.logger.info(f'Loaded {len(urls)} URLs from {self.links_path}, {len(new_urls)} not yet downloaded')
        for url in new_urls:
            yield scrapy.Request(url, callback=self.parse)

    def url_to_filename(self, url: str) -> str:
        path = urlparse(url).path.strip('/')
        slug = path.replace('/', '__') or 'index'
        if len(slug) > 150:
            digest = hashlib.sha1(url.encode()).hexdigest()[:10]
            slug = f'{slug[:140]}__{digest}'
        return f'{slug}.html'

    def parse(self, response: Response) -> Iterator[dict[str, Any]]:
        # persistence happens in HtmlWriterPipeline, the spider only extracts
        yield {
            'url': response.url,
            'file': self.url_to_filename(response.url),
            'status': response.status,
            'body': response.body,
        }
