"""Guards on the reranker's fixed 1024-token input budget."""

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rag.config import Settings
from rag.reranking import _MAX_LENGTH, _PREFIX, _SUFFIX, DEFAULT_INSTRUCTION

PROMPT_DIR = Path(__file__).resolve().parents[1] / 'eval' / 'rerank_prompts'
CHUNKS = Settings().chunks_path

# the longest eval query so the guard covers the worst realistic pair
LONGEST_QUERY = "how does a barbarian's fatigue after rage interact with fatigued condition already in effect"


def _prompts() -> list[tuple[str, str]]:
    named = [(p.stem, p.read_text(encoding='utf-8').strip()) for p in sorted(PROMPT_DIR.glob('*.txt'))]
    return [('DEFAULT_INSTRUCTION', DEFAULT_INSTRUCTION), *named]


@pytest.fixture(scope='module')
def tokenizer() -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(Settings().reranker_model, padding_side='left'))


@pytest.fixture(scope='module')
def longest_chunk() -> str:
    if not CHUNKS.exists():
        pytest.skip(f'{CHUNKS} not built')
    df = pd.read_parquet(CHUNKS, columns=['n_tokens', 'text'])
    return str(df.text.loc[df.n_tokens.idxmax()])


def _budget(tokenizer: PreTrainedTokenizerBase) -> int:
    prefix = len(tokenizer.encode(_PREFIX, add_special_tokens=False))
    suffix = len(tokenizer.encode(_SUFFIX, add_special_tokens=False))
    return _MAX_LENGTH - prefix - suffix


@pytest.mark.gpu
@pytest.mark.parametrize(('name', 'instruction'), _prompts(), ids=[n for n, _ in _prompts()])
def test_prompt_fits_budget_against_longest_chunk(
    name: str, instruction: str, tokenizer: PreTrainedTokenizerBase, longest_chunk: str
) -> None:
    pair = f'<Instruct>: {instruction}\n<Query>: {LONGEST_QUERY}\n<Document>: {longest_chunk}'
    used = len(tokenizer.encode(pair, add_special_tokens=False))
    budget = _budget(tokenizer)
    assert used <= budget, (
        f'{name}: worst-case pair is {used} tokens against a budget of {budget} '
        f'({used - budget} over). The chunk end is being truncated away — raise _MAX_LENGTH, '
        f'shorten the instruction, or lower chunk_max_tokens.'
    )


@pytest.mark.gpu
def test_chunk_cap_leaves_room_for_the_longest_prompt(tokenizer: PreTrainedTokenizerBase) -> None:
    settings = Settings()
    longest = max((instr for _, instr in _prompts()), key=lambda s: len(tokenizer.encode(s)))
    overhead = len(tokenizer.encode(f'<Instruct>: {longest}\n<Query>: {LONGEST_QUERY}\n<Document>: '))
    assert settings.chunk_max_tokens + overhead <= _budget(tokenizer), (
        f'chunk_max_tokens={settings.chunk_max_tokens} plus {overhead} tokens of instruction and '
        f'query exceeds the reranker budget of {_budget(tokenizer)}'
    )
