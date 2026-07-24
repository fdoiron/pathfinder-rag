from typing import Any

from pathfinder_scraper.spiders.d20pfsrd import D20pfsrdSpider


class HtmlWriterPipeline:
    """
    persist each page's raw HTML to disk.
    """

    def process_item(self, item: dict[str, Any], spider: D20pfsrdSpider) -> dict[str, Any]:
        filepath = spider.output_dir / item['file']
        filepath.write_bytes(item.pop('body'))
        return item
