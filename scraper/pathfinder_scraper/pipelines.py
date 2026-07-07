class HtmlWriterPipeline:
    """
    persist each page's raw HTML to disk.
    """

    def process_item(self, item, spider):
        filepath = spider.output_dir / item['file']
        filepath.write_bytes(item.pop('body'))
        return item
