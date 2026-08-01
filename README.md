[![CI](https://github.com/fdoiron/pathfinder-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/fdoiron/pathfinder-rag/actions/workflows/ci.yml)

A retrieval augmented question/answering pipeline over the Pathfinder 1e tabletop ruleset. It scrapes ~24k rule pages from [d20pfsrd.com](https://www.d20pfsrd.com/), parses them into markdown with a hand written converter, chunks and embeds them locally, and serves cited answers through a `rag ask` CLI backed by a local LLM. This is a portfolio project to demonstrate a full pipeline: data ingestion, evaluated retrieval, and generation, end to end, with the service/container phases designed and staged as what's next.

## Demo

```
$ uv run rag search "power attack"
--- Result 1 ---
Score: 0.033
Title: Power Attack (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/power-attack-combat
--- Result 2 ---
Score: 0.031
Title: Two-Handed Fighter
URL: https://www.d20pfsrd.com/classes/core-classes/fighter/archetypes/paizo-fighter-archetypes/two-handed-fighter
--- Result 3 ---
Score: 0.031
Title: Furious Focus (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/furious-focus-combat
--- Result 4 ---
Score: 0.031
Title: Mythic Tiberolith
URL: https://www.d20pfsrd.com/bestiary/monster-listings/constructs/mythic-tiberolith
--- Result 5 ---
Score: 0.029
Title: Chakra Power (Akashic)
URL: https://www.d20pfsrd.com/magic/variant-magic-rules/akashic-magic/feats/chakra-power-akashic
```

`search` defaults to `--method hybrid` so what is printed is a Reciprocal Rank Fusion score (bounded by `2/(rrf_k+1) ≈ 0.033` at `rrf_k=60`) not a similarity score. RRF ranks and doesn't measure; only the order matters and close ties like results 2 to 4 are how RRF behaves. Use `--method vector` for cosine similarities instead.

```
$ uv run rag ask "can I move and attack in the same round?"
Yes, you can move and attack in the same round by using the Spring Attack feat for melee attacks or
Shot on the Run for ranged attacks, both of which allow movement and an attack as part of a
full-round action [3][5]. Normally, without such feats, you cannot move before and after an attack
[5], but these abilities specifically grant that capability.

[3] Spring Attack (Combat) — Spring Attack (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/spring-attack-combat
[5] Shot on the Run (Combat) — Shot on the Run (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/shot-on-the-run-combat
```

Both runs above against the real corpus with the current hybrid index, `qwen3:14b` served by a local `ollama serve`. `ask` output is not deterministic, the wording and the subset of excerpts cited will vary between runs.

## What works right now

Scraping, parsing, chunking, embedding, hybrid search (BM25 + vector, fused with Reciprocal Rank Fusion) and generation all run end to end over the full corpus. `rag build-corpus` does parse, chunk, embed and build the BM25 index in one pass, writing two parquet artifacts, a SQLite FTS5 index, and a manifest; `rag search`, `rag ask` and `rag evaluate` all take `--method {vector,bm25,hybrid}` and load straight from disk to answer or score from the terminal. Containers and an API are not built yet, they're staged in [Future expansions](#future-expansions).

24,080 HTML files in, 23,890 cleaned articles out, chunked into 129,361 chunks and embedded at 1024 dims. Parsing and chunking run in under a minute single threaded; embedding the full corpus locally takes roughly 15 minutes on an RTX3090. HTML scraping with Scrapy (/scraper) takes roughly 6 hours with a 1s crawl delay per page.

Dropped pages:
- ~180 pages too small after stripping to be worth indexing
- 1 page where the original URL was too long and was hashed

```
$ cd rag
$ uv run rag build-corpus ../scraper/data/html
INFO:root:Parsing HTML files from ../scraper/data/html
WARNING:rag.parsing:dropped 'bestiary__monster-listings__aberrations__dark-young': body too short (19 < 100 chars)
INFO:root:Loading tokenizer Qwen/Qwen3-Embedding-0.6B
INFO:root:Chunking articles
wrote 23890 articles to data/corpus.parquet
INFO:root:Loading embedder Qwen/Qwen3-Embedding-0.6B
INFO:root:Embedding 129361 chunks
wrote 129361 chunks to data/chunks.parquet
wrote fts5 index to data/chunks.fts5.db
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

# 5. evaluate retrieval against the hand verified truth set
uv run rag evaluate eval/queries.jsonl

# 6. run the test suite (doesn't require the scraped corpus)
uv run pytest
```

## Repo layout

- `scraper/` : Scrapy spider + URL discovery, scrapes d20pfsrd.com into `scraper/data/html/`
- `rag/` : parsing, chunking, embedding, retrieval, eval and the `rag` CLI; everything downstream of the scraped HTML

## Architecture

**Current state**, everything is a batch step or a CLI call, no services of its own, the only long running process is Ollama:

```
scraper (Scrapy)  ──▶  scraper/data/html/ (24,080 files)
                              │
                              ▼
          rag build-corpus  (parse → chunk → embed → index)
                              │
                              ▼
    data/: corpus.parquet (23,890 articles), chunks.parquet (129,361 chunks + embeddings),
           chunks.fts5.db (SQLite FTS5 lexical index), manifest
                              │
      rag search "query" in process embedder + numpy cosine over chunks, BM25 over the
                          FTS5 index, the two rankings fused with RRF
                              │
              rag ask "question" search → prompt → Ollama (localhost, OpenAI compatible)
                              │
                    cited answer + d20pfsrd URLs
```

**Target service architecture**, staged as future work, see [Future expansions](#future-expansions):

```
┌─────────────┐     REST      ┌──────────────┐    REST (OpenAI-compat)   ┌─────────────┐
│  frontend   │ ────────────▶│   rag-api    │ ────────────────────────▶│  llm        │
│ (Streamlit/ │               │ (FastAPI +   │                           │ (vLLM or    │
│  Chainlit)  │               │  MCP server) │        REST               │  Ollama)    │
└─────────────┘               │              │ ────────────┐             └─────────────┘
                              └──────────────┘             ▼
                                    ▲              ┌─────────────┐
                             reads  │              │  embedder   │
                             corpus │              │ (TEI)       │
                             volume │              └─────────────┘
┌─────────────┐   writes     ┌──────┴───────┐
│  scraper    │ ───────────▶│ data volume  │   scraper runs as a batch Job,
│ (batch Job) │              │ (parquet +   │   NOT a long running service
└─────────────┘              │  manifests)  │
                             └──────────────┘
```

## Evaluation

34 hand verified queries in `rag/eval/queries.jsonl`, split into three types (`exact_name`, `paraphrase`, `rules_reasoning`), scored at the URL level (a document counts as found if any of its chunks lands in the top k). Vector-only vs BM25-only vs hybrid (RRF, `rrf_k=60`), all three scored at `k=5` against the same truth set and the same corpus:

| Type | Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| **Overall** (n=34) | Vector | 0.44 | 0.76 | 0.82 | 0.59 |
| | BM25 | 0.21 | 0.29 | 0.29 | 0.25 |
| | **Hybrid** | **0.53** | 0.71 | 0.79 | **0.63** |
| `exact_name` (n=11) | Vector | 0.55 | 1.00 | 1.00 | 0.73 |
| | BM25 | 0.55 | 0.82 | 0.82 | 0.68 |
| | **Hybrid** | **0.82** | 0.82 | 0.91 | **0.84** |
| `paraphrase` (n=12) | Vector | 0.25 | 0.42 | 0.50 | 0.34 |
| | BM25 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Hybrid | 0.25 | 0.42 | 0.50 | 0.34 |
| `rules_reasoning` (n=11) | Vector | 0.55 | 0.91 | 1.00 | 0.73 |
| | BM25 | 0.09 | 0.09 | 0.09 | 0.09 |
| | Hybrid | 0.55 | 0.91 | 1.00 | 0.75 |

**Note**: With n=34 queries one query is worth ~3 points overall and ~8-9 points per type. A single query flipping moves a cell more than most differences between cells in the table. The table ranks interventions against each other on the same queries. It is not a benchmark and the absolute numbers should not be read as precise. Growing the truth set to 60-100 queries in `E2` below will partially address this.

BM25 ties vector on `exact_name` recall@1 but returns nothing on `paraphrase` (MRR 0.00, a paraphrase by construction shares few literal words with its target page) and barely helps `rules_reasoning`. Fused via RRF (rrf_k=60), `exact_name` recall@1 rises to 0.82, above either individual method. RRF ranks a doc highly if it ranks well in either list, and a doc ranking first in both (an exact-name match BM25 does well on while vector gets somewhat close on) dominates the fused order. `paraphrase` is unchanged: BM25 contributes nothing there, so RRF reduces to vector's own ranking. Net change: overall MRR 0.59 → 0.63, recall@1 0.44 → 0.53. `paraphrase` staying weak either way is a chunk size problem rather than a retrieval-method problem: 51.4% of chunks are under 100 tokens, 20.6% under 50. Hence `E1.5`.

The gains from hybrid cost recall@3 (0.76 → 0.71) and recall@5 (0.82 → 0.79). Six queries moved between the two runs: four up and two down. `cleave` 2 → 1, `rage powers` 3 → 1, `summon monster I` 3 → 1, and `can you full attack after a charge` 3 → 2 all climb to or toward the top of the rankings, while `grapple combat maneuver` slips 3 → 5 and `outsider creature type` falls out of the top 5 entirely. The last one is a textbook example of RRF failure: mediocre in both lists beats excellent in one but not the other way around. The Outsiders page is the second best from vector and doesn't appear anywhere in BM25's 50 results, so the fused score is `1/(60+2) ≈ 0.016`. Meanwhile, four `Creature (CR +N)` template pages are in the middle of both lists (Cold Creature at vector 11 and BM25 8, `1/71 + 1/68 ≈ 0.029`) and each individually outscore it, taking fused ranks 1, 2, 4, and 5. Since RRF sees only rank and not score, vector's own verdict that the Outsiders page belongs nine places above those templates never reaches the fused ordering. Re-scoring the fused top 20 with a cross encoder (`E1.6`) is the targeted fix for the queries this loses.

Full per-query results and per-category breakdowns are in the run files themselves: [vector](rag/eval/canonical/n34_vector_post-glue-fix.json), [BM25](rag/eval/canonical/n34_bm25.json), [hybrid](rag/eval/canonical/n34_hybrid.json). `rag evaluate --method {vector,bm25,hybrid}` writes a new timestamped run under `eval/runs/` every time it's run; that directory is gitignored scratch space. Runs worth keeping as evidence get copied into `eval/canonical/` with a descriptive name instead of a timestamp. Before/after evidence for bug fixes that also moved the eval numbers are in [Post mortem (draft)](#post-mortem-draft) below.

## Roadmap

MVP milestones, all completed:

| Milestone | Status | Content |
|---|---|---|
| M1: parsing | **Done** | HTML → cleaned markdown `Article`s, short-page filtering, golden-file + invariant test suite |
| M2: chunking | **Done** | Heading aware section splitting, token budget packing with overlap |
| M3: local embeddings | **Done** | In process `sentence-transformers` embedder (Qwen3-Embedding-0.6B), replaces the old Vertex-only path |
| M4: search CLI | **Done** | Cosine search over chunk embeddings, thin CLI wrapper over `retrieval.py` |
| M5: eval | **Done** | 34 hand verified typed queries, Recall@k / MRR per query type and category |
| M6: `rag ask` | **Done** | Retrieval augmented generation via local Ollama server, cited answers |
| M7: README | **Done** | this file |

Future expansions:

| Expansion | Status | Content |
|---|---|---|
| E1: hybrid retrieval | **Done** | BM25 (SQLite FTS5) + vector, combined by Reciprocal Rank Fusion. Overall MRR 0.59 → 0.63, `exact_name` recall@1 0.55 → 0.82. No movement in `paraphrase` see [Evaluation](#evaluation). |
| E2: expand evaluation set | — | Grow the truth set to 60-100 queries, add answer level checks (does the answer actually cite what it retrieved).|
| E1.5: small-to-big retrieval | — | Embed small chunks for sharp search, hand the generator the whole parent section, so `max_tokens` stops having to serve both retrieval precision and generation completeness at once. More promising improvement on `paraphrase` than more BM25/RRF tuning would be |
| E1.6: reranker | — | Split out of E1's original stretch goal. Cross-encoder re-scores just the RRF top-20 |
| E3: TEI embedding service | — | Swap the in process embedder for a served one (Hugging Face's text-embeddings-inference), needed once a long running API is doing the embedding instead of a batch job |
| E4: rag-api | — | FastAPI wrapping `search`/`ask` as `POST /v1/search` and `POST /v1/ask`, plus an MCP adapter over the same functions, `/healthz` vs `/readyz` |
| E4.5: agentic ask | — | Hand the model `search` as a tool it can call itself, for multi-hop questions that need more than one retrieval pass |
| E5: vLLM swap | — | Production grade serving in place of Ollama, quantization and VRAM budgeting on a single GPU |
| E6: frontend | — | Thin Streamlit/Chainlit UI calling rag-api |
| E7: containers | — | docker compose, then K3s, scraper and embed step become Jobs instead of services |

## Design decisions

- **Hand written HTML to markdown converter.** The element vocabulary is small (19 tags), but d20pfsrd.com's stat block markup needs custom handling, hence no off the shelf library. For example, `p.title` and `p.divider` are visual conventions for section headings rather than semantic HTML and rowspan/colspan tables need padding to render rectangular.
- **Markdown is the canonical text.** After parsing, no downstream process uses the raw HTML again. Chunking, embedding and display all operate on `body_md`.
- **`doc_id` = filename slug**, which is stable. The `url` is reconstructed from the slug (`__` → `/`) rather than stored twice to avoid drift. See `_slug_to_url`.
- **Drop filters log why a page was dropped.** `parse_corpus_dir` splits drops into two distinguishable reasons (parse error / too short) logged with slug and reason. This means that if the final article count looks wrong the cause can be established with a `grep` instead of re-running with print statements.
- **Golden-file testing.** The 15 fixtures are hand picked pages and have a committed expected output file (`rag/tests/fixtures/goldens/*.golden.md`). When the parser changes, the golden file diffs the behavior change line by line. A silent regression shows up as an unintended diff instead of passing quietly.
- **Heading aware chunking, packed to a token budget.** Sections split on the markdown headings from parsing, then get packed into chunks around `max_tokens ≈ 450`. Packing works on whole units, ie a line -> that line's sentences -> raw token windows, so a break can only land *between* units and a list marker can't be separated from its body. `overlap ≈ 50` tokens is therefore conditional: the trailing unit carries into the next chunk only when a whole one fits the allowance, so a section built from large paragraphs gets none. Overlap softens a mid-thought cut, and cuts now land on boundaries. Tokens are counted with the embedder's own tokenizer, not chars/4, stat blocks are abbreviation dense and blow a char based estimate. Each chunk's text is prefixed with title and heading path, without that prefix all 8,370 bestiary DEFENSE sections look nearly identical to the embedder.
- **`max_tokens` is a hyperparameter.** It has to optimize two conflicting goals : retrieval wants small tight chunks (sharp vector) while generation wants complete rules (a fragment invites the LLM to fill gaps confidently, which is where citation backed hallucinations come from). 450 is a starting value to be measured against later (E2). `max_tokens`/`overlap` are in `Settings`, not as constants in `chunking.py` and the manifest records what `max_tokens`/`overlap` a specific `chunks.parquet` was built with.
- **In process local embedder (`sentence-transformers`, Qwen3-Embedding-0.6B) instead of a TEI server.** Corpus embedding is a batch job either way. `Embedder` is a Protocol (`rag/src/rag/models.py`), the retriever depends on "anything with an `.embed()` method", not on which implementation, so tests run without a GPU (`FakeEmbedder`) and the Vertex→local swap impacted one file. TEI will be added once a long running API needs query embeddings in E3.
- **`task_type` is the shared embedding vocabulary.** Qwen3 expects an instruction prefix on queries and nothing on documents. `LocalEmbedder` maps `task_type` to that convention in one line.
- **Brute force numpy over chunk embeddings, no vector database.** ~130k chunks at 1024 dims is ~530 MB of float32, a matrix vector product over that runs well under a millisecond. A vector DB buys index structures like HNSW that pay off at a scale this corpus isn't at. Revisit if eval or corpus size says otherwise.
- **`doc_id`/URL level eval truth, not at chunk level.** `queries.jsonl` is hand made and expensive to build. If it referenced chunk ids, every re-chunk would invalidate it. Retrieval returns chunks, but hits get collapsed to their document URL before scoring, so re-chunking, re-embedding, or swapping the model never touches the truth file.
- **Manifest guards serving.** `ChunksManifest` records the embedding model, dimension, parser version, chunk params and sha256 of the corpus parquet the chunks were produced from, next to `chunks.parquet`. `load_retriever` refuses to load if the configured settings don't match what's on disk or if the corpus has been rebuilt since (hash drift). A mismatched index fails at load time instead of quietly returning garbage scores.
- **Thin CLI over plain functions.** `rag search`, `rag evaluate` and `rag ask` parse args, call one internal function (`retriever.search`, `evaluate_query`, `answer_question`), and print. No logic lives in the typer layer. Same functions the future FastAPI handlers (E4) will call.
- **Citation resolution is ours, not the model's.** `answer_question` builds the numbered excerpt list itself. `[n]` in the model's reply maps back to position `n` in that list, so a citation can never point at a document that wasn't retrieved. The model can still cite a real excerpt that doesn't support its claim, that's what an answer level eval (E2) would catch.
- **Single shot retrieval, not agentic tool use.** `rag ask` runs exactly one search per question instead of handing the model `search` as a tool it calls itself. Agentic retrieval helps multi hop questions but turns one retrieval into a variable number of calls, which breaks the clean attribution the eval harness depends on. Staged as E4.5, once E4 exposes an MCP tool surface to hang it off of.
- **BM25 in SQLite FTS5.** The lexical index is a file (`chunks.fts5.db`) written by `build-corpus` next to `chunks.parquet`. It avoids having an Elasticsearch/OpenSearch process to run, configure and keep alive for a CLI that is otherwise all batch steps. FTS5 comes with the standard library's `sqlite3`. Being a file means it joins the artifact set the manifest already checks: `load_retriever` refuses to start without it the same way it refuses a stale corpus hash.
- **RRF instead of weighted score fusion.** A cosine similarity in `[-1, 1]` and an FTS5 `bm25()` score (negative, unbounded, corpus dependent) share no scale. `a*vector + b*bm25` needs a normalization step that has to be fitted, and refitted whenever the corpus changes. RRF discards the scores and fuses positions instead, `1/(rrf_k + rank)` summed across lists. No normalization or tuning is required beyond `rrf_k`. The cost is "how far ahead" gets discarded along with the scale which is the regression documented in [Evaluation](#evaluation).
- **Title is its own FTS5 column and weighted 10:1 over body text.** The index stores `heading_path[0]` as a `title` column separate from `text`, so `bm25()` can score a hit in the page's name above the same word buried in its body. `exact_name` queries are a page title by definition. A feat page's body repeats the generic combat vocabulary of every other feat page. A landmine: `bm25()` takes one weight per column in declared order, including `UNINDEXED` ones. Skipping a slot shifts every following weight onto the wrong column and fails silently instead of erroring.
- **The MATCH string is rebuilt from tokens.** `_sanitize_match_query` extracts word tokens and quotes each one. User text is not FTS5 text: a trailing `?`, a bare `-`, or an unbalanced `"` are syntax and raise `sqlite3.OperationalError`, and an uppercase `OR`/`AND`/`NOT` sitting in an ordinary question is read as an operator instead of a word, which fails silently. Raw, `move OR attack` matches 22,595 chunks; quoted into `"move" "or" "attack"` it matches 1,186. The tokenizer flag it splits on comes from `manifest.fts5_tokenchar`, not `Settings`, since `tokenchars '-'` is fixed into the index at `CREATE VIRTUAL TABLE` time and a query tokenized differently than the index would miss. The bm25 weights come from `Settings`. They apply at query time and are meant to be tuned. The manifest records them as provenance only.

## Testing

`ruff check`, `ruff format --check`, and `mypy` in strict mode, 350+ tests. The parsing suite covers golden file tests over 15 fixtures, invariant tests parametrized on the 15 fixtures (no unescaped HTML, no license boilerplate, rendered table's rows match its header's column count), and unit tests for every converter rule, heading retagging edge case, and drop filter reason. Chunking, embedding, retrieval, eval and `answer_question` are all tested by faking the boundary they touch (a fake tokenizer/`SentenceTransformer`/OpenAI chat client), no GPU, no network, no downloaded weights required to run the suite. One real end to end test per GPU dependent module is marked `@pytest.mark.gpu` and skipped by default.

```bash
cd rag
uv run poe check   #in order:  ruff check ., ruff format --check ., mypy src tests, pytest
```

## Post mortem (draft)

### Bugs

**Chunking: lists and paragraphs glued into prose.** Sections over `max_tokens` came back with bullets glued onto one line, sometimes with a marker ending one chunk and its own text starting the next. The sentence splitter cut on `\s+`, which eats newlines, and the packer rejoined with `' '.join(...)` with no record a newline was ever there. Fixed by packing whole units (line, then sentences, then raw token windows) instead of sentences, each rejoined with its own original separator, so a break can no longer land inside a marker's body.

Hidden because the only test on chunk text used `.split()` to compare word sequences, which discards whitespace entirely. `- Alertness.\n- Dodge.` and `- Alertness. - Dodge.` were the same value to it: the test encoded what the code did (preserve words), not what it should (preserve structure). Any normalization inside an assertion (`.split()`, `.strip()`, `.lower()`, sorting) is a blind spot with that exact shape. New test asserts every source line comes back whole in some chunk, no normalization involved.

Eval: [before](rag/eval/canonical/n34_vector_pre-chunking-fix.json) → [after](rag/eval/canonical/n34_vector_post-chunking-fix.json). Recall@5 0.76 → 0.82.

**Parser: flavor text glued onto headings.** ~88 pages wrap flavor text in a bare `<span>`/`<i>`/`<b>` instead of `<p class="description">`, so the block separator never fired and the text concatenated onto the heading line. Fixed by tracking the previous sibling's tag and forcing a break after any heading, regardless of the next child's own tag.

Hidden because the converter's separator logic covered the common markup pattern and nothing checked for pages that didn't follow it. No test asserted "a heading is always followed by a break" as its own property, only cases derived from the 15 golden fixtures, none of which happened to hit this pattern.

Eval: [before](rag/eval/canonical/n34_vector_post-chunking-fix.json) → [after](rag/eval/canonical/n34_vector_post-glue-fix.json). `exact_name` recall@1 0.45 → 0.55.

## License and attribution

This repo's code is licensed under [Apache License 2.0](LICENSE). The content it parses is Open Game Content from d20pfsrd.com, itself drawn from Paizo's Pathfinder Roleplaying Game and a substantial amount of third-party OGL publishers, released under the [Open Game License v1.0a](LICENSE-OGL.txt). See [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md) for the full attribution, including a (programmatically generated) list of sourcebooks cited. Pathfinder is a trademark of Paizo Inc. This is an unaffiliated fan/portfolio project.
