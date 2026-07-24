A retrieval augmented question/answering pipeline over the Pathfinder 1e tabletop ruleset. It scrapes ~24k rule pages from [d20pfsrd.com](https://www.d20pfsrd.com/), parses them into markdown with a hand written converter, chunks and embeds them locally, and serves cited answers through a `rag ask` CLI backed by a local LLM. This is a portfolio project to demonstrate a full pipeline: data ingestion, evaluated retrieval, and generation, end to end, with the service/container phases designed and staged as what's next.

## Demo

```
$ uv run rag search "power attack"
--- Result 1 ---
Score: 0.570
Title: Power Attack (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/power-attack-combat
--- Result 2 ---
Score: 0.551
Title: Furious Focus (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/furious-focus-combat
--- Result 3 ---
Score: 0.534
Title: Two-Handed Fighter
URL: https://www.d20pfsrd.com/classes/core-classes/fighter/archetypes/paizo-fighter-archetypes/two-handed-fighter
```

```
$ uv run rag ask "can I move and attack in the same round?"
Yes, you can move and attack in the same round using specific full-round actions. For example, the
Spring Attack feat allows you to move up to your speed, make a single melee attack, and move again
as part of a full-round action, without provoking attacks of opportunity [2]. Similarly, Shot on the
Run allows moving, firing a ranged attack, and moving again as a full-round action [4]. Normally,
movement and attacks are separate actions, but these abilities explicitly combine them.

Additionally, during a full-attack action, you can choose to take a move action instead of completing
your remaining attacks, though this typically applies to melee combatants after the first attack [1].
However, standard movement and attack actions (not combined) require separate actions unless modified
by feats or abilities.

[1] Combat — Combat > Actions In Combat > Full-Round Actions > Full Attack > Deciding between an Attack or a Full Attack
    https://www.d20pfsrd.com/gamemastering/combat
[2] Spring Attack (Combat) — Spring Attack (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/spring-attack-combat
[4] Shot on the Run (Combat) — Shot on the Run (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/shot-on-the-run-combat
```

Both runs above against the real corpus, `qwen3:14b` served by a local `ollama serve`.

## What works right now

Scraping, parsing, chunking, embedding, search and generation all run end to end over the full corpus. `rag build-corpus` does parse, chunk and embed in one pass and writes two parquet artifacts plus a manifest; `rag search` and `rag ask` load them and answer from the terminal. Containers, an API and hybrid retrieval are not built yet, they're staged in [Future expansions](#future-expansions).

24,080 HTML files in, 23,890 cleaned articles out, chunked into 129,200 chunks and embedded at 1024 dims. Parsing and chunking run in under a minute single threaded; embedding the full corpus locally takes roughly 15 minutes on an RTX3090. HTML scraping with Scrapy (/scraper) takes roughly 6 hours with a 1s crawl delay per page.

Dropped pages:
- ~180 pages too small after stripping to be worth indexing
- 1 page where the original URL was too long and was hashed

```
$ cd rag
$ uv run rag build-corpus ../scraper/data/html
INFO:root:Parsing HTML files from ../scraper/data/html
WARNING:rag.parsing:dropped 'bestiary__monster-listings__aberrations__dark-young': body too short (19 < 100 chars)
INFO:rag.parsing:dropped 'classes': hub page
INFO:rag.parsing:dropped 'equipment': hub page
INFO:root:Loading tokenizer Qwen/Qwen3-Embedding-0.6B
INFO:root:Chunking articles
wrote 23890 articles to data/corpus.parquet
INFO:root:Loading embedder Qwen/Qwen3-Embedding-0.6B
INFO:root:Embedding 129200 chunks
wrote 129200 chunks to data/chunks.parquet
wrote manifest to data/chunks.manifest.json
```

## Quickstart

The scraped HTML corpus is not included in the repo (24k files). You need to run the scraper first, or point `build-corpus` at your own directory of d20pfsrd.com HTML pages. `rag ask` also needs a local Ollama server with a model pulled.

```bash
git clone https://github.com/fdoiron/pathfinder-rag.git
cd pathfinder-rag

# 1. scrape (slow, respects a 1s crawl delay, ~24k pages)
cd scraper
uv sync
cp .env.example .env # fill in a contact email for the User-Agent
uv run python discover_urls.py # discovers and filters the URL list to scrape, writes d20pfsrd_links.parquet
uv run scrapy crawl d20pfsrd

# 2. parse, chunk and embed into a searchable corpus
cd ../rag
uv sync
uv run rag build-corpus ../scraper/data/html

# 3. search (no LLM needed, only the embedding model)
uv run rag search "power attack"

# 4. ask, needs a local Ollama server
ollama pull qwen3:14b
ollama serve
uv run rag ask "can I move and attack in the same round?"

# 5. run the test suite (doesn't require the scraped corpus)
uv run pytest
```

## Architecture

**Current state**, everything is a batch step or a CLI call, no services of its own, the only long running process is Ollama:

```
scraper (Scrapy)  ──▶  scraper/data/html/ (24,080 files)
                              │
                              ▼
              rag build-corpus  (parse → chunk → embed)
                              │
                              ▼
        data/: corpus.parquet (23,890 articles), chunks.parquet (129,200 chunks + embeddings), manifest
                              │
              rag search "query"        in process embedder + numpy cosine over chunks
                              │
              rag ask "question"        search → prompt → Ollama (localhost, OpenAI compatible)
                              │
                    cited answer + d20pfsrd URLs
```

**Target service architecture**, staged as future work, see [Future expansions](#future-expansions):

```
┌─────────────┐     REST      ┌──────────────┐    REST (OpenAI-compat)   ┌─────────────┐
│  frontend   │ ────────────▶ │   rag-api    │ ────────────────────────▶ │  llm        │
│ (Streamlit/ │               │ (FastAPI +   │                           │ (vLLM or    │
│  Chainlit)  │               │  MCP server) │        REST               │  Ollama)    │
└─────────────┘               │              │ ────────────┐             └─────────────┘
                              └──────────────┘             ▼
                                     ▲              ┌─────────────┐
                              reads  │              │  embedder   │
                              corpus │              │ (TEI)       │
                              volume │              └─────────────┘
┌─────────────┐   writes     ┌──────┴───────┐
│  scraper    │ ───────────▶ │ data volume  │   scraper runs as a batch Job,
│ (batch Job) │              │ (parquet +   │   NOT a long running service
└─────────────┘              │  manifests)  │
                             └──────────────┘
```

## Evaluation

34 hand verified queries in `rag/eval/queries.jsonl`, split into three types (`exact_name`, `paraphrase`, `rules_reasoning`), scored at the URL level (a document counts as found if any of its chunks lands in the top k). Latest run, vector only, `k=5`:

| | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| **Overall** (n=34) | 0.41 | 0.74 | 0.76 | 0.57 |
| `exact_name` (n=11) | 0.55 | 0.91 | 0.91 | 0.70 |
| `paraphrase` (n=12) | 0.25 | 0.42 | 0.50 | 0.34 |
| `rules_reasoning` (n=11) | 0.45 | 0.91 | 0.91 | 0.68 |

Reading these: `exact_name` and `rules_reasoning` both do well, a named feat or a rules question about combat tends to embed close to the page that answers it. `paraphrase` is the weak point, worded around the name rather than using it ("spell that throws an exploding ball of fire" for fireball) is exactly where a vector only search struggles and where lexical/BM25 matching is expected to help, see `E1` below. Per category breakdowns and the full per query results are in the run file itself, `rag evaluate` writes a new timestamped one under `eval/runs/` every time it's run.

## Roadmap

MVP milestones, all completed:

| Milestone | Status | Content |
|---|---|---|
| M1: parsing | **Done** | HTML → cleaned markdown `Article`s, hub/short-page filtering, golden-file + invariant test suite |
| M2: chunking | **Done** | Heading aware section splitting, token budget packing with overlap |
| M3: local embeddings | **Done** | In process `sentence-transformers` embedder (Qwen3-Embedding-0.6B), replaces the old Vertex-only path |
| M4: search CLI | **Done** | Cosine search over chunk embeddings, thin CLI wrapper over `retrieval.py` |
| M5: eval | **Done** | 34 hand verified typed queries, Recall@k / MRR per query type and category |
| M6: `rag ask` | **Done** | Retrieval augmented generation via local Ollama server, cited answers |
| M7: README | **Done** | this file |

Future expansions:

| Expansion | Content |
|---|---|
| E1: hybrid retrieval | BM25 (SQLite FTS5) + vector, combined by Reciprocal Rank Fusion, reranker as a stretch. Targets the `paraphrase` weak point above  |
| E1.5: small-to-big retrieval | Embed small chunks for sharp search, hand the generator the whole parent section, so `max_tokens` stops having to serve both retrieval precision and generation completeness at once |
| E2: expand evaluation set | Grow the truth set to 60-100 queries, add answer level checks (does the answer actually cite what it retrieved) |
| E3: TEI embedding service | Swap the in process embedder for a served one (Hugging Face's text-embeddings-inference), needed once a long running API is doing the embedding instead of a batch job |
| E4: rag-api | FastAPI wrapping `search`/`ask` as `POST /v1/search` and `POST /v1/ask`, plus an MCP adapter over the same functions, `/healthz` vs `/readyz` |
| E4.5: agentic ask | Hand the model `search` as a tool it can call itself, for multi-hop questions that need more than one retrieval pass |
| E5: vLLM swap | Production grade serving in place of Ollama, quantization and VRAM budgeting on a single GPU |
| E6: frontend | Thin Streamlit/Chainlit UI calling rag-api |
| E7: containers | docker compose, then K3s, scraper and embed step become Jobs instead of services |

## Design decisions

- **Hand written HTML to markdown converter.** The element vocabulary is small (19 tags), but d20pfsrd.com's stat block markup needs custom handling, hence no off the shelf library. For example, `p.title` and `p.divider` are visual conventions for section headings rather than semantic HTML and rowspan/colspan tables need padding to render rectangular.
- **Markdown is the canonical text.** After parsing, no downstream process uses the raw HTML again. Chunking, embedding and display all operate on `body_md`.
- **`doc_id` = filename slug**, which is stable. The `url` is reconstructed from the slug (`__` → `/`) rather than stored twice to avoid drift. See `_slug_to_url`.
- **Drop filters log why a page was dropped.** `parse_corpus_dir` splits drops into three distinguishable reasons (hub page / too short / parse error) at different log levels. This means that if the final article count looks wrong the cause can be established with a `grep` instead of re-running with print statements.
- **Golden-file testing.** The 15 fixtures are hand picked pages and have a committed expected output file (`rag/tests/fixtures/goldens/*.golden.md`). When the parser changes, the golden file diffs the behavior change line by line. A silent regression shows up as an unintended diff instead of passing quietly.
- **Heading aware chunking, packed to a token budget.** Sections split on the markdown headings from parsing, then get packed into chunks around `max_tokens ≈ 450` with `overlap ≈ 50` tokens so a boundary straddling sentence is whole somewhere. Tokens are counted with the embedder's own tokenizer, not chars/4, stat blocks are abbreviation dense and blow a char based estimate. Each chunk's text is prefixed with title and heading path, without that prefix all 8,370 bestiary DEFENSE sections look nearly identical to the embedder.
- **`max_tokens` is a hyperparameter.** It sits in a real tension: retrieval wants small tight chunks (sharp vector) while generation wants complete rules (a fragment invites the LLM to fill gaps confidently, which is where citation backed hallucinations come from). 450 is a starting value chosen to be measured against later (E2), not a final decision. `max_tokens`/`overlap` are in `Settings`, not as literals in `chunking.py`, and the manifest records what a given `chunks.parquet` was built with.
- **In process local embedder (`sentence-transformers`, Qwen3-Embedding-0.6B) instead of a TEI server.** Corpus embedding is a batch job either way. `Embedder` is a Protocol (`rag/src/rag/models.py`), the retriever depends on "anything with an `.embed()` method", not on which implementation, so tests run without a GPU (`FakeEmbedder`) and the Vertex→local swap impacted one file. TEI will be added once a long running API needs query embeddings in E3.
- **`task_type` is the shared embedding vocabulary.** Qwen3 expects an instruction prefix on queries and nothing on documents. `LocalEmbedder` maps `task_type` to that convention in one line.
- **Brute force numpy over chunk embeddings, no vector database.** ~130k chunks at 1024 dims is ~530 MB of float32, a matrix vector product over that runs well under a millisecond. A vector DB buys index structures like HNSW that pay off at a scale this corpus isn't at. Revisit if eval or corpus size says otherwise.
- **`doc_id`/URL level eval truth, not at chunk level.** `queries.jsonl` is hand made and expensive to build. If it referenced chunk ids, every re-chunk would invalidate it. Retrieval returns chunks, but hits get collapsed to their document URL before scoring, so re-chunking, re-embedding, or swapping the model never touches the truth file.
- **Manifest guards serving.** `ChunksManifest` records the embedding model, dimension, parser version and chunk params next to `chunks.parquet`. `load_retriever` refuses to load if the configured settings don't match what's on disk. A mismatched index fails at load time instead of quietly returning garbage scores.
- **Thin CLI over plain functions.** `rag search`, `rag evaluate` and `rag ask` parse args, call one internal function (`retriever.search`, `evaluate_query`, `answer_question`), and print. No logic lives in the typer layer. Same functions the future FastAPI handlers (E4) will call.
- **Citation resolution is ours, not the model's.** `answer_question` builds the numbered excerpt list itself. `[n]` in the model's reply maps back to position `n` in that list, so a citation can never point at a document that wasn't retrieved. The model can still cite a real excerpt that doesn't support its claim, that's what an answer level eval (E2) would catch.
- **Single shot retrieval, not agentic tool use.** `rag ask` runs exactly one search per question instead of handing the model `search` as a tool it calls itself. Agentic retrieval helps multi hop questions but turns one retrieval into a variable number of calls, which breaks the clean attribution the eval harness depends on. Staged as E4.5, once E4 exposes an MCP tool surface to hang it off of.

## Testing

`ruff check`, `ruff format --check`, and `mypy --strict`, 357 tests. The parsing suite covers golden file tests over 15 fixtures, invariant tests parametrized on the 15 fixtures (no unescaped HTML, no license boilerplate, rendered table's rows match its header's column count), and unit tests for every converter rule, heading retagging edge case, and drop filter reason. Chunking, embedding, retrieval, eval and `answer_question` are all tested by faking the boundary they touch (a fake tokenizer/`SentenceTransformer`/OpenAI chat client), no GPU, no network, no downloaded weights required to run the suite. One real end to end test per GPU dependent module is marked `@pytest.mark.gpu` and skipped by default.

```bash
cd rag
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

## License and attribution

This repo's code is licensed under [Apache License 2.0](LICENSE). The content it parses is Open Game Content from d20pfsrd.com, itself drawn from Paizo's Pathfinder Roleplaying Game and a substantial amount of third-party OGL publishers, released under the [Open Game License v1.0a](LICENSE-OGL.txt). See [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md) for the full attribution, including a (programmatically generated) list of sourcebooks cited. Pathfinder is a trademark of Paizo Inc. This is an unaffiliated fan/portfolio project.
