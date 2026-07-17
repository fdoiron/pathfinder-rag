A retrieval augmented question/answering pipeline over the Pathfinder 1e tabletop ruleset. It scrapes ~24k rule pages from [d20pfsrd.com](https://www.d20pfsrd.com/), parses them into markdown with a hand written converter, and (once finished) will chunk, embed and serve them through a `rag search` CLI with a local LLM. This is a portfolio project to deploy a full data ingestion to evaluated retrieval to (once finished) a served API.

## What works right now

Scraping and parsing are done and tested, run over the full corpus. Chunking, embedding, search and generation are not yet built (see [Roadmap](#roadmap)). This README describes the current state of the repo.

24,080 HTML files in, 23,863 cleaned articles out. Run time is ~43s single-threaded. HTML scraping with Scrapy (/scraper) takes roughly 6 hours with 1s delay per page.
Dropped pages :
- 9 category pages/link lists (see [`is_hub_page`](rag/src/rag/parsing.py))
- 207 pages too small after stripping to be worth indexing
- 1 page where the original URL was too long and was hashed.

```
$ cd rag
$ uv run rag build-corpus ../scraper/data/html
INFO:root:Parsing HTML files from ../scraper/data/html
WARNING:rag.parsing:dropped 'bestiary__monster-listings__aberrations__dark-young': body too short (19 < 100 chars)
WARNING:rag.parsing:dropped 'bestiary__monster-listings__aberrations__naga': body too short (7 < 100 chars)
INFO:rag.parsing:dropped 'classes': hub page
INFO:rag.parsing:dropped 'equipment': hub page
INFO:rag.parsing:dropped 'feats': hub page
WARNING:rag.parsing:dropped 'classes__core-classes__sorcerer__archetypes__paizo-sorcerer-archetypes__wildblooded__mutated-bloodlines-legendary-games__new-wildblooded-sor__c35fd7cde8': parse error: Slug '...' was truncated and hashed by url_to_filename
...
wrote 23863 articles to data/corpus.parquet
```

A rough Vertex based prototype of embed/search/eval exists at [`rag/scripts/prototype_cli.py`](rag/scripts/prototype_cli.py), with tests over its `embedding.py`/`retrieval.py`/`evaluation.py` modules, but it is not installed as part of `rag` and will be replaced by the local-embedding pipeline described in the roadmap below.

## Quickstart

The scraped HTML corpus is not included in the repo (24k files). You need to run the scraper first, or point `build-corpus` at your own directory of d20pfsrd.com HTML pages.

```bash
git clone https://github.com/fdoiron/pathfinder-rag.git
cd pathfinder-rag

# 1. scrape (slow, respects a 1s crawl delay. ~24k pages)
cd scraper
uv sync
cp .env.example .env # fill in a contact email for the User-Agent
uv run python discover_urls.py # discovers and filters the URL list to scrape, writes d20pfsrd_links.parquet
uv run scrapy crawl d20pfsrd

# 2. parse into a cleaned corpus
cd ../rag
uv sync
uv run rag build-corpus ../scraper/data/html

# 3. run the test suite (doesn't require the scraped corpus)
uv run pytest
```


## Architecture

**Current state** single batch step with no services:

```
scraper (Scrapy)  ──▶  scraper/data/html/ (24,080 files)
                              │
                              ▼
                 rag build-corpus  (parse_corpus_dir)
                              │
                              ▼
                   data/corpus.parquet  (23,863 articles)
```

**CLI Target**  chunking, local embedding, vector search, and generation via a local LLM, still all batch/CLI:

```
scraper/data/html/ ──▶ parse ──▶ chunk ──▶ embed ──▶ chunks.parquet
                                                          │
                                          rag search "query" ──▶ ranked results
                                                          │
                                          rag ask "question" ──▶ Ollama (local)
                                                          │
                                                cited answer + d20pfsrd URLs
```

**Architecture Target** wrap search/ask functions in FastAPI service, swap in process embedder for TEI container, add hybrid BM25+vector retrieval and containerize for K3s. See [Roadmap](#roadmap)


## Roadmap

| Milestone | Status | Content |
|---|---|---|
| M1 : parsing | **Done** | HTML → cleaned markdown `Article`s, hub/short-page filtering, golden-file + invariant test suite  |
| M2 : chunking | Designed | Heading-aware section splitting, token budget packing with overlap |
| M3 : local embeddings | Designed | In process `sentence-transformers` embedder (Qwen3-Embedding-0.6B) to replace the current Vertex-only path |
| M4 : search CLI | Designed | Cosine search over chunk embeddings. Thin CLI wrapper over a `retrieval.py` function |
| M5 : eval | Designed | ~30 hand-verified typed queries, Recall@k / MRR per query type and category |
| M6 : `rag ask` | Designed | Retrieval-augmented generation via local Ollama server, cited answers |
| M7 : service/containers | Designed | FastAPI + MCP adapters, TEI, hybrid retrieval, docker compose & K3s |



## Design decisions

- **Hand written HTML to markdown converter**. The element vocabulary is small (19 tags), but d20pfsrd.com's stat block markup needs custom handling, hence no off the shelf library. For example, `p.title` and `p.divider` are visual conventions for section headings rather than semantic HTML and rowspan/colspan tables need padding to render rectangular.
- **Markdown is the canonical text**. After parsing, no downstream process uses the raw HTML again. Chunking, embedding and display all operate on `body_md`.
- **`doc_id` = filename slug** which is stable. The `url` is reconstructed from the slug (`__` → `/`) rather than stored twice to avoid drift. See `_slug_to_url`.
- **Drop filters log why a page was dropped**. `parse_corpus_dir` splits drops into three distinguishable reasons (hub page / too short / parse error) at different log levels. This ensures that if the final article count looks wrong the cause can be established with a `grep` instead of re-running with print statements.
- **Golden-file testing**. The 15 fixtures are hand picked pages and have a committed expected output file (`rag/tests/fixtures/goldens/*.golden.md`). When the parser changes, the golden file diffs documents the behavior change line by line. A silent regression shows up as an unintended diff instead of passing quietly.
- **Thin CLI over plain functions**. `rag build-corpus` calls `parse_corpus_dir` and writes parquet with no logic in the typer layer. The same function is what the planned API handler would call.
- **`max_tokens` is a target with small token slack**. The packer counts tokens by summing each piece's token counts to decide if a window is full. However, the tokenizer does it across one string, which can cause the total to be more than the sum of its parts. On the 24k pages corpus, the drift happens 8 times out of ~128k chunks, with a total of 451 vs. a target of 450. I deemed the small loss of retrieval precision acceptable, but something to be aware of. At max_tokens = 1000, there were zero chunks over the limit.

## Testing

`ruff check`, `ruff format --check`, and `mypy`. The parsing suite covers: golden file tests over 15 fixtures, invariant tests parametrized on the 15 fixtures (no unescaped HTML, no license boilerplate, rendered table's rows match its header's column count), and unit tests for every converter rule, heading retagging edge case, and drop filter reason.

```bash
cd rag
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

## License and attribution

This repo's code is licensed under [Apache License 2.0](LICENSE). The content it parses is Open Game Content from d20pfsrd.com, itself drawn from Paizo's Pathfinder Roleplaying Game and a substantial amount of third-party OGL publishers, released under the [Open Game License v1.0a](LICENSE-OGL.txt). See [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md) for the full attribution, including a (programmatically generated) list of sourcebooks cited. Pathfinder is a trademark of Paizo Inc. This is an unaffiliated fan/portfolio project.