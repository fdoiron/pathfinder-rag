[![CI](https://github.com/fdoiron/pathfinder-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/fdoiron/pathfinder-rag/actions/workflows/ci.yml)

A retrieval augmented question/answering pipeline over the Pathfinder 1e tabletop ruleset. It scrapes ~24k rule pages from [d20pfsrd.com](https://www.d20pfsrd.com/), parses them into markdown with a hand written converter, chunks and embeds them locally, and serves cited answers through a `rag ask` CLI backed by a local LLM. This is a portfolio project to demonstrate a full pipeline: data ingestion, evaluated retrieval, and generation, end to end, with the service/container phases designed and staged as what's next.

## Demo

```
$ uv run rag search "power attack"
--- Result 1 ---
Score: 0.262
Title: Power Attack (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/power-attack-combat
--- Result 2 ---
Score: 0.257
Title: Furious Focus (Combat)
URL: https://www.d20pfsrd.com/feats/combat-feats/furious-focus-combat
--- Result 3 ---
Score: 0.254
Title: Two-Handed Fighter
URL: https://www.d20pfsrd.com/classes/core-classes/fighter/archetypes/paizo-fighter-archetypes/two-handed-fighter
--- Result 4 ---
Score: 0.250
Title: Mythic Tiberolith
URL: https://www.d20pfsrd.com/bestiary/monster-listings/constructs/mythic-tiberolith
--- Result 5 ---
Score: 0.244
Title: Powerful Poisoning
URL: https://www.d20pfsrd.com/feats/general-feats/powerful-poisoning
```

`search` defaults to `--method hybrid` so what is printed is a Reciprocal Rank Fusion score (bounded by `(rrf_vector_weight+rrf_bm25_weight)/(rrf_k+1) ≈ 0.262` at the current defaults, `rrf_k=60`, `rrf_vector_weight=15`, `rrf_bm25_weight=1`) not a similarity score. RRF ranks and doesn't measure; only the order matters and close scores like results 2 to 4 are how RRF behaves. Use `--method vector` for cosine similarities instead. `--method bm25` prints SQLite FTS5's raw `bm25()` score instead: negative and unbounded, more negative ranks better.

```
$ uv run rag ask "can I move and attack in the same round?"
Yes, you can move and attack in the same round using specific abilities. For example, the **Spring
Attack** feat allows you to move up to your speed, make a single melee attack, and move again as
part of a full-round action [3]. Similarly, **Shot on the Run** enables moving, firing a ranged
attack, and moving again as a full-round action [5]. These feats explicitly allow movement and
attacks within the same round.

Additionally, during a **full-attack action**, you may choose to take a move action instead of
making remaining attacks after your first attack [1].

The retrieved excerpts cover this.

[1] Combat — Combat > Actions In Combat > Full-Round Actions > Full Attack > Deciding between an Attack or a Full Attack
    https://www.d20pfsrd.com/gamemastering/combat
[3] Spring Attack (Combat) — Spring Attack (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/spring-attack-combat
[5] Shot on the Run (Combat) — Shot on the Run (Combat)
    https://www.d20pfsrd.com/feats/combat-feats/shot-on-the-run-combat
```

Both runs above against the real corpus with the current hybrid index, `qwen3:14b` served by a local `ollama serve`. `ask` output is not deterministic, the wording and the subset of excerpts cited will vary between runs.

The `ask` answer above is also a live example of a current gap : every excerpt it retrieved is feat- or full-attack-specific (Spring Attack, Shot on the Run, two full-attack sub-sections of the combat page), so the model frames the whole answer around feats rather than the baseline rule that a move action plus a standard attack needs no feat at all. Retrieval never found that baseline chunk, so the model had nothing correct to cite and leaned on what it had instead. See [Evaluation](#evaluation) for the broader pattern of confident answers with incomplete retrieval behind them.

## What works right now

Scraping, parsing, chunking, embedding, hybrid search (BM25 + vector, fused with Reciprocal Rank Fusion) and generation all run end to end over the full corpus. `rag build-corpus` does parse, chunk, embed and build the BM25 index in one pass, writing two parquet artifacts, a SQLite FTS5 index, and a manifest; `rag search`, `rag ask` and `rag evaluate` all take `--method {vector,bm25,hybrid}` and load straight from disk to answer or score from the terminal. `rag evaluate-answers` runs the same truth set through `rag ask` itself, scoring whether the generated answer cites the correct source, invents a citation number outside what it was given, or refuses when it doesn't know. Containers and an API are not built yet, they're staged in [Future expansions](#future-expansions).

24,098 HTML files in, 23,890 cleaned articles out, chunked into 129,361 chunks and embedded at 1024 dims. Parsing and chunking run in under a minute single threaded; embedding the full corpus locally takes roughly 15 minutes on an RTX3090. HTML scraping with Scrapy (/scraper) takes roughly 6 hours with a 1s crawl delay per page.

Dropped pages (208 total):
- 207 pages too small after stripping to be worth indexing
- 1 page where the original URL was too long and was hashed so the source URL couldn't be reconstructed

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

The scraped HTML corpus is not included in the repo (24k files). You need to run the scraper first, or point `build-corpus` at your own directory of d20pfsrd.com HTML pages. `rag ask` also needs a local Ollama server with a model pulled. Whichever command uses the embedder first (`build-corpus` or `search`) downloads ~1.2 GB of Qwen3-Embedding-0.6B weights from Hugging Face Hub as a one time cost. Cached afterwards.

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

# 6. evaluate whether rag ask's answers actually cite the correct source
uv run rag evaluate-answers eval/queries.jsonl

# 7. run the test suite (doesn't require the scraped corpus)
uv run pytest
```

## Repo layout

- `scraper/` : Scrapy spider + URL discovery, scrapes d20pfsrd.com into `scraper/data/html/`
- `rag/` : parsing, chunking, embedding, retrieval, eval and the `rag` CLI; everything downstream of the scraped HTML

## Architecture

**Current state**, everything is a batch step or a CLI call, no services of its own, the only long running process is Ollama:

```
scraper (Scrapy)  ──▶  scraper/data/html/ (24,098 files)
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

127 hand verified queries in `rag/eval/queries.jsonl`, split into three types (`exact_name`, `paraphrase`, `rules_reasoning`), scored at the URL level (a document counts as found if any of its chunks lands in the retrieved set). Vector-only, BM25-only and hybrid (RRF, `rrf_k=60`, vector weighted 15x over BM25 in the fused score) were each run with a retrieval depth of 5 against the same truth set and the same corpus:

| Type | Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| **Overall** (n=127) | Vector | 0.45 | 0.73 | 0.80 | 0.60 |
| | BM25 | 0.22 | 0.31 | 0.35 | 0.27 |
| | **Hybrid** | **0.46** | 0.75 | 0.80 | **0.61** |
| `exact_name` (n=55) | Vector | 0.51 | 0.78 | 0.87 | 0.66 |
| | BM25 | 0.49 | 0.67 | 0.76 | 0.59 |
| | **Hybrid** | **0.55** | 0.82 | 0.89 | **0.69** |
| `paraphrase` (n=38) | Vector | 0.32 | 0.53 | 0.55 | 0.42 |
| | BM25 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Hybrid | 0.32 | 0.53 | 0.55 | 0.42 |
| `rules_reasoning` (n=34) | Vector | 0.50 | 0.88 | 0.94 | 0.69 |
| | BM25 | 0.03 | 0.06 | 0.06 | 0.04 |
| | Hybrid | 0.50 | 0.88 | 0.94 | 0.69 |


BM25 ties vector on `exact_name` recall@1 (0.49 vs 0.51) but returns nothing on `paraphrase` (MRR 0.00, a paraphrase by construction shares few literal words with its target page) and is not useful on `rules_reasoning` (recall@1 0.03). Fusing it in at equal (1:1) RRF weight was a net negative at n=127. recall@3, recall@5 and overall MRR all moved against hybrid for only a noise level recall@1 tick. A close sibling or variant of the correct page (a poison stat block sharing a condition's name, a `Greater X` version of the base feat, a parent class page over its own subpage) routinely outscored the canonical page on literal term overlap and RRF has no sense of how close it is numerically, it only uses ranks, so it cannot tell that apart from a genuinely wrong page. Weighting vector 15x over BM25 in the fused score (swept in `scripts/rrf_weight_sweep.py`, [sweep output](rag/eval/canonical/n127_rrf_weight_sweep.json)) fixes this issue. That value was picked by maximizing MRR over this same n=127 set, with no held out split at this size to sharpen it further. It sits on a plateau rather than a knife's edge: weights 10, 15, 20, 30 all score MRR 0.603-0.609, a spread smaller than one query flipping. The defensible claim is heavily down-weight BM25 not that 15 specifically is optimal.

Down-weighting BM25 contains its failure mode rather than eliminating it entirely. 18 queries still change rank between vector-only and weighted hybrid (10 up, 8 down), about half the churn of the unweighted version (32, split 15/17), and the same sibling page confusion from above is still visible in the 8 moving down. `outsider creature type` (2 → 3) still loses to general reference pages (`Simple Monster Creation`, `Creature Types & Subtypes`) ahead of the specific `Outsiders` category page, and `sorcerer bloodline` (2 → 3) still loses to the parent `Sorcerer` class page over its own `Bloodlines` subpage. The difference is severity. Under unweighted RRF, `fatigued`/`exhausted`/`shaken condition` all fell from a clean vector rank 1 to a complete miss, beaten outright by short poison/drug entries that happen to name the condition once (`Nerveblast` for `shaken condition`). Weighted, the same three queries only slip to rank 2-4. `shaken condition` still loses to `Nerveblast` at rank 1, but the correct `Conditions` page now lands at rank 4, still inside recall@5. BM25 keeps enough influence to rescue real ties (`cleave`, `dodge feat`, `improved critical`, `point blank shot`, and `ranger favored enemy` all climb to rank 1) without enough voting power left to drag a lexically similar wrong page above vector's correct pick outright. Re-scoring the fused top 20 with a cross encoder (`E1.6`) remains the next lever for the residual 8. The cross encoder reads content, not just term overlap, so it's the more plausible fix for cases weighting alone contains but doesn't fully resolve.

Full per-query results and per-category breakdowns are in the run files themselves: [vector](rag/eval/canonical/n127_vector.json), [BM25](rag/eval/canonical/n127_bm25.json), [hybrid](rag/eval/canonical/n127_hybrid.json), and the unweighted (1:1) hybrid run the paragraph above compares against: [hybrid, unweighted](rag/eval/canonical/n127_hybrid_unweighted.json). `rag evaluate --method {vector,bm25,hybrid}` writes a new timestamped run under `eval/runs/` every time it's run; that directory is gitignored scratch space. Runs worth keeping as evidence get copied into `eval/canonical/` with a descriptive name instead of a timestamp. Before/after evidence for bug fixes that also moved the eval numbers are in [Post mortem (draft)](#post-mortem-draft) below.

`rag evaluate-answers` runs the same truth set through `answer_question` instead of raw retrieval scoring four things per query: whether the correct source was even retrieved into the prompt, whether the answer cited it, whether it invented a citation number outside the excerpts it was given, and whether it honestly refused rather than answering uncited. Hybrid, weighted RRF (vector 15x over BM25), `ask_k=5` (the same default `rag ask` currently serves):

| Type | n | Retrieved | Cited | Refused | Invented citation |
|---|---|---|---|---|---|
| **Overall** | 127 | 0.80 | 0.71 | 0.10 | 0.00 |
| `exact_name` | 55 | 0.87 | 0.82 | 0.05 | 0.00 |
| `paraphrase` | 38 | 0.55 | 0.53 | 0.13 | 0.00 |
| `rules_reasoning` | 34 | 0.94 | 0.74 | 0.15 | 0.00 |

`Invented citation` is a narrow, structural check: did any `[n]` resolve to something outside the excerpts the model was actually given. Results show zero across all 127 queries, but it is not a general hallucination check. The numbers right below it show why that distinction matters. Retrieval improved along with the RRF reweighting: 101 of 127 queries (80%, up from 75% under unweighted hybrid) now get the correct source into the 5 excerpts handed to the LLM. Citation quality tracks this. Of those 101, 90 (89.1%, up from 86.3%) were cited correctly. Of the 26 queries where retrieval still comes up empty, only 7 (27%, up from 16%) say the excerpts don't cover it. The other 19 (73%) answer confidently anyway with nothing correct behind them. Those are real hallucinations, in the ordinary sense of the word, just not the kind `Invented citation` is built to catch. While there are fewer of them now than before there is roughly 3 confident wrong answers for every honest refusal, which is the same ratio as before. Cutting into those 19 is scoped as `E2.5` as a prompt-side fix.

Five queries are still fully silent (retrieved, not cited, didn't refuse) a similar count to before (was four), though better retrieval resolved some of the old ones and identified different ones: `undead creature type`, `class ability where a monk gets extra unarmed attacks`, and three multipart rules questions (`can you full attack after a charge`, `do two-weapon fighting penalties stack with power attack`, `what happens if you're grappled and try to cast a spell`). Of the 6 queries refused despite the correct source being present, 2 still share the broad `gamemastering/combat` umbrella page (`does concealment stack with cover`, `caster level at negative HP`), the same URL-level blind spot as before (a big page landing one irrelevant chunk in the top 5 counts as "retrieved" even when the specific needed section isn't there), just smaller now that retrieval itself improved.

The demo at the top of this README was one of the silent misses under unweighted hybrid. Under the current default it correctly cites the canonical `gamemastering/combat` page for `can I move and attack in the same round?`, a concrete, visible effect of the RRF reweighting. (Generation is non-deterministic, see the note under [Demo](#demo), so a re-run may cite differently, but the underlying retrieval now consistently shows the right page where it previously didn't.)

Full per-query results: [answer eval, hybrid k=5, weighted RRF](rag/eval/canonical/n127_answer_eval_hybrid.json) — the "up from" figures above compare against the same run under unweighted (1:1) RRF: [answer eval, hybrid k=5, unweighted RRF](rag/eval/canonical/n127_answer_eval_hybrid_unweighted.json).

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
| E1: hybrid retrieval | **Done** | BM25 (SQLite FTS5) + vector, combined by Reciprocal Rank Fusion. At n=127, `exact_name` recall@1 0.51 → 0.55, MRR 0.60 → 0.61. No movement in `paraphrase` see [Evaluation](#evaluation). |
| E2: expand evaluation set | **Done** | Truth set grown to 127 queries (incl. real play session questions). Answer level checks built and run (`rag evaluate-answers`): 80% retrieval, 71% correctly cited, 0% invented citations but only 27% honest refusal when retrieval actually fails. 73% answer confidently with nothing correct behind them, real hallucinations the invented citation check cannot see. See [Evaluation](#evaluation). |
| E2.5: reduce confident hallucination | — | 19 of 26 retrieval failure queries answer confidently anyway instead of refusing (see [Evaluation](#evaluation)). Prompt-side fix (stronger/repeated refusal instruction, an explicit "is this actually covered" check before answering). |
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
- **RRF instead of weighted score fusion.** A cosine similarity in `[-1, 1]` and an FTS5 `bm25()` score (negative, unbounded, corpus dependent) share no scale. `a*vector + b*bm25` needs a normalization step that has to be fitted, and refitted whenever the corpus changes. RRF discards the scores and fuses positions instead, `1/(rrf_k + rank)` summed across lists. No normalization or tuning is required beyond `rrf_k`. The cost is "how far ahead" gets discarded along with the scale which is the regression documented in [Evaluation](#evaluation). Fusion runs over a pool of `max(k, 50)` candidates per list, so the fused order is itself a function of `k`: a page outside the 50-candidate pool at `k=5` can enter the pool at `k=100` and change ranks even inside the top 5.
- **Title is its own FTS5 column and weighted 10:1 over body text.** The index stores `heading_path[0]` as a `title` column separate from `text`, so `bm25()` can score a hit in the page's name above the same word buried in its body. `exact_name` queries are a page title by definition. A feat page's body repeats the generic combat vocabulary of every other feat page. A landmine: `bm25()` takes one weight per column in declared order, including `UNINDEXED` ones. Skipping a slot shifts every following weight onto the wrong column and fails silently instead of erroring.
- **The MATCH string is rebuilt from tokens.** `_sanitize_match_query` extracts word tokens and quotes each one. User text is not FTS5 text: a trailing `?`, a bare `-`, or an unbalanced `"` are syntax and raise `sqlite3.OperationalError`, and an uppercase `OR`/`AND`/`NOT` sitting in an ordinary question is read as an operator instead of a word, which fails silently. Raw, `move OR attack` matches 22,595 chunks; quoted into `"move" "or" "attack"` it matches 1,186. The tokenizer flag it splits on comes from `manifest.fts5_tokenchar`, not `Settings`, since `tokenchars '-'` is fixed into the index at `CREATE VIRTUAL TABLE` time and a query tokenized differently than the index would miss. The bm25 weights come from `Settings`. They apply at query time and are meant to be tuned. The manifest records them as provenance only.

## Testing

`ruff check`, `ruff format --check`, and `mypy` in strict mode, 500+ tests. The parsing suite covers golden file tests over 15 fixtures, invariant tests parametrized on the 15 fixtures (no unescaped HTML, no license boilerplate, rendered table's rows match its header's column count), and unit tests for every converter rule, heading retagging edge case, and drop filter reason. Chunking, embedding, retrieval, eval and `answer_question` are all tested by faking the boundary they touch (a fake tokenizer/`SentenceTransformer`/OpenAI chat client), no GPU, no network, no downloaded weights required to run the suite. One real end to end test per GPU dependent module is marked `@pytest.mark.gpu` and skipped by default.

```bash
cd rag
uv run poe check   #in order:  ruff check ., ruff format --check ., mypy src tests, pytest
```

## Post mortem (draft)

### Eval methodology notes

**The original evaluation table was n=34.** One query was worth about 3 points overall and 8 to 9 points per type. A single query flipping moved a cell more than most differences between cells in the table. The table ranked interventions against each other on the same queries, but it was not a benchmark and the absolute numbers were not precise. The [Evaluation](#evaluation) section's per-bug before/after deltas below predate the regrow and should be read with that in mind. Growing the truth set to 127 queries (`E2`) fixed this. At n=127 one query is worth well under a point overall (about 0.8) and 2 to 3 points per type, close enough to trust the table at face value.

### Bugs

**Chunking: lists and paragraphs glued into prose.** Sections over `max_tokens` came back with bullets glued onto one line, sometimes with a marker ending one chunk and its own text starting the next. The sentence splitter cut on `\s+`, which eats newlines, and the packer rejoined with `' '.join(...)` with no record a newline was ever there. Fixed by packing whole units (line, then sentences, then raw token windows) instead of sentences, each rejoined with its own original separator, so a break can no longer land inside a marker's body.

Hidden because the only test on chunk text used `.split()` to compare word sequences, which discards whitespace entirely. `- Alertness.\n- Dodge.` and `- Alertness. - Dodge.` were the same value to it: the test encoded what the code did (preserve words), not what it should (preserve structure). Any normalization inside an assertion (`.split()`, `.strip()`, `.lower()`, sorting) is a blind spot with that exact shape. New test asserts every source line comes back whole in some chunk, no normalization involved.

Eval: [before](rag/eval/canonical/n34_vector_pre-chunking-fix.json) → [after](rag/eval/canonical/n34_vector_post-chunking-fix.json). Recall@5 0.76 → 0.82.

**Parser: flavor text glued onto headings.** ~88 pages wrap flavor text in a bare `<span>`/`<i>`/`<b>` instead of `<p class="description">`, so the block separator never fired and the text concatenated onto the heading line. Fixed by tracking the previous sibling's tag and forcing a break after any heading, regardless of the next child's own tag.

Hidden because the converter's separator logic covered the common markup pattern and nothing checked for pages that didn't follow it. No test asserted "a heading is always followed by a break" as its own property, only cases derived from the 15 golden fixtures, none of which happened to hit this pattern.

Eval: [before](rag/eval/canonical/n34_vector_post-chunking-fix.json) → [after](rag/eval/canonical/n34_vector_post-glue-fix.json). `exact_name` recall@1 0.45 → 0.55.

## License and attribution

This repo's code is licensed under [Apache License 2.0](LICENSE). The content it parses is Open Game Content from d20pfsrd.com, itself drawn from Paizo's Pathfinder Roleplaying Game and a substantial amount of third-party OGL publishers, released under the [Open Game License v1.0a](LICENSE-OGL.txt). See [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md) for the full attribution, including a (programmatically generated) list of sourcebooks cited. Pathfinder is a trademark of Paizo Inc. This is an unaffiliated fan/portfolio project.
