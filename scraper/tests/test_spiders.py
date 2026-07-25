import pytest

from pathfinder_scraper.spiders.d20pfsrd import D20pfsrdSpider


def _make_spider(tmp_path) -> D20pfsrdSpider:
    return D20pfsrdSpider(links_path='unused.parquet', output_dir=tmp_path)


def test_url_to_filename_slugs_path_segments(tmp_path):
    spider = _make_spider(tmp_path)
    url = 'https://www.d20pfsrd.com/bestiary/monster-listings/aberrations/aboleth/'
    assert spider.url_to_filename(url) == 'bestiary__monster-listings__aberrations__aboleth.html'


def test_url_to_filename_root_path_is_index(tmp_path):
    spider = _make_spider(tmp_path)
    assert spider.url_to_filename('https://www.d20pfsrd.com/') == 'index.html'


def test_url_to_filename_long_slug_gets_hashed(tmp_path):
    spider = _make_spider(tmp_path)
    url = 'https://www.d20pfsrd.com/' + 'a' * 200
    filename = spider.url_to_filename(url)
    assert len(filename) < len(url)
    assert filename.endswith('.html')


@pytest.mark.parametrize('url', ['https://www.d20pfsrd.com/x/', 'https://www.d20pfsrd.com/x'])
def test_url_to_filename_ignores_trailing_slash(tmp_path, url):
    spider = _make_spider(tmp_path)
    assert spider.url_to_filename(url) == 'x.html'
