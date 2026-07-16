"""
Discover the set of d20pfsrd.com URLs to scrape and write them to d20pfsrd_links.parquet.

Reads the site's sitemap, then applies hand-tuned per-category include/exclude
rules to keep official Paizo content and drop lists, tools, and third-party
publisher pages.

Re-run this whenever the URL list needs refreshing; d20pfsrd_links.parquet
is gitignored, so 'scrapy crawl d20pfsrd' needs this to have been run at least once.

Rerunning may pick up new/removed pages since the site changes over time. The exact
counts in the README were captured from one specific run.
"""

import os
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv
from lxml import etree

load_dotenv()

UA = os.getenv('SCRAPER_CONTACT_UA')
if not UA:
    raise RuntimeError(
        'SCRAPER_CONTACT_UA is not set. Copy .env.example to .env and fill in a contact '
        'User-Agent before running discover_urls.py.'
    )
HEADERS = {'User-Agent': UA}
DELAY = 1.0

THIRD_PARTY_PUBLISHERS = [
    '3rd-party-publisher',
    '3rd-party-publishers',
    '4-winds-fantasy-gaming',
    'adamant-entertainment',
    'ascension-games',
    'ascension-games-llc',
    'azoth-games',
    'bloodstone-press',
    'd20pfsrd-com-publishing',
    'dreamscarred-press',
    'drop-dead-studios',
    'everyman-gaming',
    'everyman-gaming-llc',
    'far-distant-future-publishing',
    'fat-goblin-games',
    'fire-mountain-games',
    'flaming-crab-games',
    'flying-pincushion-games',
    'forest-guardian-press',
    'frog-god-games',
    'icosa-entertainment-llc',
    'jon-brazer-enterprises',
    'jon-brazer-enterprizes',
    'kobold-press',
    'kobold-press-open-design-llc',
    'legendary-games',
    'little-red-goblin-games',
    'little-red-goblin-games-llc',
    'louis-porter-jr-design',
    'michael-mars',
    'misfit-studios',
    'names-games',
    'necromancers-of-the-northwest',
    'onyx-path-and-nocturnal-publishing',
    'orphaned-bookworm-productions',
    'paizo-fans-united',
    'petersen-games',
    'purple-duck-games',
    'radiance-house',
    'rite-publishing',
    'rogue-genius-games',
    'samurai-sheepdog',
    'shm-publishing',
    'spes-magna-games',
    'studio-m',
    'the-knotty-works',
    'total-party-kill-games',
    'varyags-forge',
    'xoth-net-publishing',
    'james-ray',
    'librarians-leviathans',
    'arcanist-exploits-3rd-party',
]

THIRD_PARTY_TERMS = [
    '3rd-party',
    '3pp',
    '4-winds-fantasy-gaming',
]

THIRD_PARTY_PATTERN = '|'.join(re.escape(p) for p in THIRD_PARTY_PUBLISHERS + THIRD_PARTY_TERMS)

# top-level categories that are out of scope:
# 'work-area' (private staging), 'subscribe' (subscription advert),
# 'extras' (mostly homebrew), 'alternative-rule-systems' (scope only base rules)

BESTIARY_LEVEL2_EXCLUDE = [
    'bestiary-alphabetical',  # lists
    'bestiary-by-challenge-rating',  # lists
    'bestiary-by-terrain',  # lists
    'bestiary-hub',  # lists
    'fan-conversions',  # scope only official
    'indexes-and-tables',  # index monsters by CR/locations
    'tools',  # interactive tools
]

BESTIARY_URL_EXCLUDE = [
    'https://www.d20pfsrd.com/bestiary/',  # top level
    'https://www.d20pfsrd.com/bestiary/rules-for-monsters/',  # list
    'https://www.d20pfsrd.com/bestiary/rules-for-monsters/monster-roles/',  # list
    'https://www.d20pfsrd.com/bestiary/rules-for-monsters/creature-types/new-creature-subtypes/',  # third party
    'https://www.d20pfsrd.com/bestiary/rules-for-monsters/universal-monster-rules/umr-3pp-frog-god-games/',
    'https://www.d20pfsrd.com/bestiary/unique-monsters/',  # list
    'https://www.d20pfsrd.com/bestiary/unique-monsters/under-cr-1/',  # list
    'https://www.d20pfsrd.com/bestiary/npc-s/npc-db/',  # excel table
    'https://www.d20pfsrd.com/bestiary/monster-listings/',  # list of monsters by type
    'https://www.d20pfsrd.com/bestiary/monster-listings/aberrations/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/animals/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/constructs/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/dragons/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/fey/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/humanoids/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/magical-beasts/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/monstrous-humanoids/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/oozes/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/outsiders/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/plants/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/undead/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/vermin/',
    'https://www.d20pfsrd.com/bestiary/monster-listings/templates/',  # template list
    'https://www.d20pfsrd.com/classes/',  # top level
    'https://www.d20pfsrd.com/classes/core-classes/',  # list
]
BESTIARY_URL_EXCLUDE += [f'https://www.d20pfsrd.com/bestiary/npc-s/npcs-cr-{cr}/' for cr in range(1, 22)]
BESTIARY_URL_EXCLUDE += [f'https://www.d20pfsrd.com/bestiary/unique-monsters/cr-{cr}/' for cr in range(1, 26)]

CLASSES_LEVEL2_INCLUDE = [
    'core-classes',
    '',
    # '3rd-party-classes', #scope only official
    # '3rd-party-npc-classes', #scope only official
    # '3rd-party-prestige-classes', #scope only official
    'alternate-classes',
    'base-classes',
    'character-advancement',
    'class-archetypes',
    # 'arcane-archetypes-rogue-genius-games', #scope only official
    'hybrid-classes',
    'monster-classes',
    'npc-classes',
    'prestige-classes',
    'unchained-classes',
    'ex-class-archetypes',
]
CLASSES_LEVEL3_EXCLUDE = [
    '3rd-party-alternate-classes',  # scope only official
    '3rd-party-hybrid-classes',  # scope only official
    'hill-giant-monster-class',  # scope only official
    'lizardfolk-monster-class',  # scope only official
    'astral-deva',  # scope only official
]
CLASSES_LEVEL4_EXCLUDE = [
    '3rd-party-shaman-spirits',
    'arcanist-greater-exploits',
    'archetypes-bloodstone-press',
    'archetypes-orphaned-bookworm-productions-2',
    'fighter-bravery-alternative-options',
    'gods-3rd-party-publishers',
    'hexes-3rd-party-publishers',
    'unchained-barbarian-archetypes-shm-publishing',
    'vampire-hunter-archetypes-legendary-games',
    'witch-patrons-3rd-party-publishers',
]
CLASSES_LEVEL6_EXCLUDE = [
    'animal-companion-archetypes-by-other-publishers',
]

EQUIPMENT_LEVEL2_EXCLUDE = ['3rd-party-equipment', 'special-materials-third-party']

GAMEMASTERING_LEVEL2_EXCLUDE = ['tools']

MAGIC_LEVEL2_EXCLUDE = ['tools']
MAGIC_URL_EXCLUDE = [
    'https://www.d20pfsrd.com/magic/all-spells/',
    'https://www.d20pfsrd.com/magic/spell-lists-and-domains/',
    'https://www.d20pfsrd.com/magic/tools/',
    'https://www.d20pfsrd.com/magic/variant-magic-rules/',
    *(f'https://www.d20pfsrd.com/magic/all-spells/{letter}/' for letter in 'abcdefghijklmnopqrstuvwxyz'),
    'https://www.d20pfsrd.com/magic/all-spells/spells-a-d/',
    'https://www.d20pfsrd.com/magic/all-spells/spells-e-h/',
    'https://www.d20pfsrd.com/magic/all-spells/spells-i-l/',
    'https://www.d20pfsrd.com/magic/all-spells/spells-m-p/',
    'https://www.d20pfsrd.com/magic/all-spells/spells-q-t/',
    'https://www.d20pfsrd.com/magic/all-spells/spells-u-z/',
]

MAGIC_ITEMS_URL_EXCLUDE = ['https://www.d20pfsrd.com/magic-items/magic-items-db/']

SKILLS_URL_EXCLUDE = ['https://www.d20pfsrd.com/skills/skills-from-other-publishers/']

TRAITS_LEVEL2_EXCLUDE = ['campaign-traits', 'tools']


def make_session() -> requests_cache.CachedSession:
    return requests_cache.CachedSession(
        'd20pfsrd_cache',
        backend='sqlite',
        expire_after=-1,
        allowable_codes=(200, 404),  # d20pfsrd sometimes 404s with valid sitemap bodies
    )


def get(session: requests_cache.CachedSession, url: str) -> requests.Response:
    resp = session.get(url, headers=HEADERS)
    if not getattr(resp, 'from_cache', False):
        time.sleep(DELAY)
    looks_like_xml = b'<urlset' in resp.content[:500] or b'<sitemapindex' in resp.content[:500]
    if resp.status_code != 200 and not looks_like_xml:
        resp.raise_for_status()
    return resp


def get_locs(session: requests_cache.CachedSession, url: str) -> list[str]:
    resp = get(session, url)
    root = etree.fromstring(resp.content)
    return root.xpath('//*[local-name()="loc"]/text()')


def fetch_sitemap_urls(session: requests_cache.CachedSession) -> pd.DataFrame:
    index = get_locs(session, 'https://www.d20pfsrd.com/wp-sitemap.xml')
    page_sitemaps = [u for u in index if 'posts-page' in u]

    rows = []
    for url in page_sitemaps:
        match = re.search(r'page-(\d+)\.xml', url)
        if match is None:
            raise ValueError(f'sitemap URL {url!r} does not match the expected ...page-N.xml pattern')
        page_num = int(match.group(1))
        locs = get_locs(session, url)
        print(f'sitemap page {page_num}: {len(locs)} urls')
        rows.extend({'sitemap_page': page_num, 'url': loc} for loc in locs)

    df = pd.DataFrame(rows)
    path = df['url'].apply(lambda u: urlparse(u).path.strip('/'))
    segments = path.apply(lambda p: p.split('/') if p else [])
    for level in range(1, 9):
        df[f'level_{level}'] = segments.apply(lambda s, level=level: s[level - 1] if len(s) >= level else '')
    return df


def filter_urls(df: pd.DataFrame) -> pd.DataFrame:
    df_alignment_description = df[df['level_1'] == 'alignment-description'].copy()
    df_basics_ability_scores = df[df['level_1'] == 'basics-ability-scores'].copy()
    df_bestiary = df[df['level_1'] == 'bestiary'].copy()
    df_classes = df[df['level_1'] == 'classes'].copy()
    df_equipment = df[df['level_1'] == 'equipment'].copy()
    df_feats = df[df['level_1'] == 'feats'].copy()
    df_gamemastering = df[df['level_1'] == 'gamemastering'].copy()
    df_magic = df[df['level_1'] == 'magic'].copy()
    df_magic_items = df[df['level_1'] == 'magic-items'].copy()
    df_races = df[df['level_1'] == 'races'].copy()
    df_skills = df[df['level_1'] == 'skills'].copy()
    df_traits = df[df['level_1'] == 'traits'].copy()

    df_alignment_description = df_alignment_description[
        ~df_alignment_description['level_2'].isin([''])
    ]  # top level list

    df_basics_ability_scores = df_basics_ability_scores[
        ~df_basics_ability_scores['url'].isin(
            [
                'https://www.d20pfsrd.com/basics-ability-scores/',  # top level list
                'https://www.d20pfsrd.com/basics-ability-scores/more-character-options/',  # top level list
            ]
        )
    ]

    df_bestiary = df_bestiary[~df_bestiary['level_2'].isin(BESTIARY_LEVEL2_EXCLUDE)].copy()
    df_bestiary = df_bestiary[~df_bestiary['url'].isin(BESTIARY_URL_EXCLUDE)]
    mask = df_bestiary['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_bestiary = df_bestiary[~mask].copy()

    df_classes = df_classes[
        (df_classes['level_2'].isin(CLASSES_LEVEL2_INCLUDE))
        & (~df_classes['level_3'].isin(CLASSES_LEVEL3_EXCLUDE))
        & (~df_classes['level_4'].isin(CLASSES_LEVEL4_EXCLUDE))
        & (~df_classes['level_6'].isin(CLASSES_LEVEL6_EXCLUDE))
    ]
    df_classes = df_classes[~df_classes['level_5'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)].copy()

    df_equipment = df_equipment[~df_equipment['level_2'].isin(EQUIPMENT_LEVEL2_EXCLUDE)]
    mask = df_equipment['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_equipment = df_equipment[~mask].copy()

    mask = df_feats['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_feats = df_feats[~mask].copy()

    df_gamemastering = df_gamemastering[~df_gamemastering['level_2'].isin(GAMEMASTERING_LEVEL2_EXCLUDE)]
    mask = df_gamemastering['level_3'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True) | df_gamemastering[
        'level_4'
    ].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_gamemastering = df_gamemastering[~mask]

    mask = df_magic['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_magic = df_magic[~mask].copy()
    df_magic = df_magic[~df_magic['level_2'].isin(MAGIC_LEVEL2_EXCLUDE)]
    df_magic = df_magic[~df_magic['url'].isin(MAGIC_URL_EXCLUDE)]

    mask = df_magic_items['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_magic_items = df_magic_items[~mask].copy()
    df_magic_items = df_magic_items[~df_magic_items['url'].isin(MAGIC_ITEMS_URL_EXCLUDE)]

    mask = df_races['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_races = df_races[~mask].copy()

    mask = df_skills['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_skills = df_skills[~mask].copy()
    df_skills = df_skills[~df_skills['url'].isin(SKILLS_URL_EXCLUDE)]

    mask = df_traits['url'].str.contains(THIRD_PARTY_PATTERN, na=False, regex=True)
    df_traits = df_traits[~mask].copy()
    df_traits = df_traits[~df_traits['level_2'].isin(TRAITS_LEVEL2_EXCLUDE)]

    return pd.concat(
        [
            df_alignment_description,
            df_basics_ability_scores,
            df_bestiary,
            df_classes,
            df_equipment,
            df_feats,
            df_gamemastering,
            df_magic,
            df_magic_items,
            df_races,
            df_skills,
            df_traits,
        ],
        ignore_index=True,
    )


def main() -> None:
    session = make_session()
    df = fetch_sitemap_urls(session)
    df_final = filter_urls(df)

    print(f'{len(df)} urls in sitemap, {len(df_final)} kept, {len(df) - len(df_final)} dropped')
    print(df_final['level_1'].value_counts().to_string())

    df_final.to_parquet('d20pfsrd_links.parquet')
    print(f'wrote {len(df_final)} urls to d20pfsrd_links.parquet')


if __name__ == '__main__':
    main()
