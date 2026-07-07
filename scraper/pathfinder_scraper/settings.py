import os

from dotenv import load_dotenv

load_dotenv()

# Scrapy settings for pathfinder_scraper project
# Full reference: https://docs.scrapy.org/en/latest/topics/settings.html

BOT_NAME = 'pathfinder_scraper'

SPIDER_MODULES = ['pathfinder_scraper.spiders']
NEWSPIDER_MODULE = 'pathfinder_scraper.spiders'

ADDONS = {}

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = os.environ.get(
    'SCRAPER_CONTACT_UA',
    'pathfinder_scraper_portfolio (contact: set SCRAPER_CONTACT_UA env var)',
)

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Concurrency and throttling settings
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 1
AUTOTHROTTLE_ENABLED = True

# Persist raw HTML via pipeline (spider yields items, pipeline writes files)
ITEM_PIPELINES = {
    'pathfinder_scraper.pipelines.HtmlWriterPipeline': 300,
}

# HTTP caching — every fetched page is kept forever in .scrapy/httpcache so
# reruns during development never re-hit the site
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = 'utf-8'
