This project’s corpus is built from [d20pfsrd.com](https://www.d20pfsrd.com/), a fan-run aggregation of Open Game Content for the Pathfinder Roleplaying Game (1st Edition), released under the [Open Game License v1.0a](https://www.d20pfsrd.com/extras/legal/) (OGL), the full text of which is included in this repo as [`LICENSE-OGL.txt`](LICENSE-OGL.txt). None of the rules text was written by this project’s author. The content originates from Paizo Inc. / Paizo Publishing, LLC's official Pathfinder RPG line and a large amount of third-party OGL content that d20pfsrd.com includes into the same categories (bestiary, feats, etc).

This repo contains code and does not republish a copy of the source site, with one exception: the 15 HTML fixtures under `rag/tests/fixtures/HTML/` are verbatim copies of individual d20pfsrd.com pages, retained to test the HTML to markdown parser against real markup. Each fixture still carries its own section15 OGL notice and remains Open Game Content subject to the OGL, same as the rest of the site. The scraped HTML (scraper/data/html) is gitignored and the parsed corpus (data/*.parquet) is built locally and not distributed publicly. “rag build-corpus” regenerates the parsed corpus from a fresh crawl.

## Artwork

Per d20pfsrd.com's own legal terms, all artwork and illustrations on the site are purchased stock, commissioned, public-domain, or used by permission, and are not Open Game Content; they may not be reused without express written permission from d20pfsrd.com Publishing / Open Gaming, LLC. This project extracts text only.

## Section 15

Every d20pfsrd.com page has its own OGL Section 15 (div.section15) that lists the sourcebook(s) the page is drawn from. parse_page (rag/src/rag/parsing.py) removes the section15 block from body_md as it is not rules content and would be noisy for retrieval. The resulting text is still Open Game Content and still subject to the OGL.

The last crawl of 24,080 pages contains 22,199 citations which can be found in [`LICENSE-THIRD-PARTY-SOURCES.txt`](LICENSE-THIRD-PARTY-SOURCES.txt). In addition to Paizo, there is a large list of third-party OGL publishers: Green Ronin (Advanced Bestiary), Frog God Games (Rappan Athuk), Necromancer Games, Open Design, Kobold Press, and many other independent publishers. The most cited sourcebooks :

| Citations | Sourcebook |
|---:|---|
| 1,069 | Pathfinder Roleplaying Game: Ultimate Equipment |
| 825 | Advanced Player's Guide |
| 726 | Pathfinder Roleplaying Game Ultimate Combat |
| 656 | Pathfinder Roleplaying Game: Ultimate Magic |
| 632 | Pathfinder Roleplaying Game: Advanced Class Guide |
| 475 | Pathfinder Roleplaying Game Advanced Race Guide |
| 426 | Pathfinder RPG Core Rulebook |
| 419 | Pathfinder Roleplaying Game Ultimate Wilderness |
| 367 | Pathfinder Roleplaying Game Bestiary 2 |
| 350 | Pathfinder Roleplaying Game Monster Codex |
| 348 | Pathfinder Roleplaying Game NPC Codex |
| 343 | Pathfinder Roleplaying Game Bestiary 3 |
| 321 | Pathfinder Campaign Setting: Inner Sea Gods |
| 304 | Pathfinder Roleplaying Game Bestiary 5 |
| 292 | Pathfinder Roleplaying Game Ultimate Intrigue |
| 285 | Pathfinder Roleplaying Game Horror Adventures |
| 251 | Pathfinder Roleplaying Game Bestiary 6 |
| 227 | Pathfinder Roleplaying Game Adventurer's Guide |


## Attribution and Product Identity

This project claims no ownership over the Pathfinder rules text. All Open Game Content herein is used under the terms of the OGL v1.0a. Product Identity (as defined by the OGL: publisher trade dress, logos, proper nouns from specific settings, etc.) is not Open Game Content and is not knowingly reproduced here beyond what d20pfsrd.com’s own Open Game Content extraction already permits. See d20pfsrd.com’s legal page : <https://www.d20pfsrd.com/extras/legal/>.

Pathfinder is a trademark of Paizo Inc. This is an unofficial fan/portfolio project, not affiliated with or endorsed by Paizo Inc.