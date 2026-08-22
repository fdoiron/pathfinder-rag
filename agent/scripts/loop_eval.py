"""Measure how the agent loop behaves over a fixed question set.

Needs a live MCP server and a live LLM.
uv run python scripts/loop_eval.py --repeats 3
"""

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Annotated, Any, cast

import typer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import AliasChoices, BaseModel, Field

from pathfinder_agent.agent import NO_COVERAGE_REPLY, open_mcp_session, run_agent
from pathfinder_agent.config import Settings, get_settings
from pathfinder_agent.llm import make_llm_client
from pathfinder_agent.runner import leaf_causes, load_system_prompt


class EvalQuestion(BaseModel):
    """One line of a JSONL question set"""

    question: str = Field(validation_alias=AliasChoices('question', 'query'))
    type: str | None = None
    expected_urls: list[str] = []  # known correct pages when the set carries them


SEARCH_TOOL = 'rag_search'
FETCH_TOOL = 'fetch_section'
ERROR_MARKERS = ('[retryable]', '[rephrase]', '[fatal]')
URL_RE = re.compile(r'https?://[^\s<>()\[\]"\']+')


@dataclass
class Run:
    question: str
    qtype: str | None
    hops: int
    seconds: float
    tool_tokens: int
    stopped_reason: str
    called: Counter[str]
    failed: Counter[str]
    citations: int
    ungrounded: list[str]
    no_coverage: bool
    expected_hit: bool | None


def _rate(failed: int, called: int) -> str:
    return f'{failed / called:.1%}' if called else 'n/a'


def _pct(values: list[float], fraction: float) -> float:
    """Nearest rank percentile. statistics.quantiles needs n>=2 and a sweep can stop after one run"""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _unwrap(content: str) -> str:
    return content.removeprefix('<tool_result>\n').removesuffix('\n</tool_result>')


def _urls(text: str) -> set[str]:
    return {match.rstrip('.,;:!?)\'"').rstrip('/') for match in URL_RE.findall(text)}


def prompt_digest(path: Path) -> str:
    """sha256 of a prompt file so a run record says which text produced it rather than only when."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def is_failure(content: str) -> bool:
    """A tool turn that carried no usable result: an SDK exception or a classified error."""
    body = _unwrap(content)
    if 'Error executing tool' in body or any(marker in body for marker in ERROR_MARKERS):
        return True
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and bool(payload.get('error_category'))


def tool_turns(transcript: list[Any]) -> list[str]:
    return [_unwrap(m.get('content') or '') for m in transcript if m.get('role') == 'tool']


def count_tool_tokens(transcript: list[Any]) -> int:
    """Same len//4 estimate run_agent charges against agent_tool_result_token_budget."""
    return sum(len(turn) // 4 for turn in tool_turns(transcript))


def check_citations(answer: str, transcript: list[Any]) -> tuple[int, list[str]]:
    """Cited URLs and those the tools never returned. The system prompt forbids any URL of the model's own."""
    cited = _urls(answer)
    if not cited:
        return 0, []
    returned = _urls(' '.join(tool_turns(transcript)))
    return len(cited), sorted(cited - returned)


def _page(url: str) -> str:
    return url.split('#')[0].rstrip('/')


def check_expected(answer: str, expected_urls: list[str]) -> bool | None:
    """Did the answer cite a known correct page. Compared without the fragment, so a right page carrying a
    fabricated anchor still counts as found here and is charged separately by check_citations."""
    if not expected_urls:
        return None
    cited = {_page(url) for url in _urls(answer)}
    return any(_page(url) in cited for url in expected_urls)


def tally_tools(transcript: list[Any]) -> tuple[Counter[str], Counter[str]]:
    """Count calls the model asked for and how many came back with an error per tool name

    Call with no result turn from the transcript counts as called but not failed: the outcome is unknown rather than bad
    """
    names: dict[str, str] = {}
    called: Counter[str] = Counter()
    failed: Counter[str] = Counter()

    for message in transcript:
        for call in message.get('tool_calls') or []:
            names[call['id']] = call['function']['name']
            called[call['function']['name']] += 1

    for message in transcript:
        if message.get('role') == 'tool' and is_failure(message.get('content') or ''):
            failed[names.get(message.get('tool_call_id'), 'unknown')] += 1

    return called, failed


def load_questions(path: Path) -> list[EvalQuestion]:
    """Parse JSONL question set. One EvalQuestion per non empty line"""
    questions: list[EvalQuestion] = []
    for line_no, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            questions.append(EvalQuestion.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f'{path}:{line_no}: invalid eval question: {exc}') from exc
    if not questions:
        raise ValueError(f'{path}: no questions found')
    return questions


class RecordingLLM:
    """Wraps the real client so a run's transcript can be kept without run_agent having to hand it back"""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client
        self.messages: list[Any] = []
        self.replies: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def reset(self) -> None:
        self.messages = []
        self.replies = []

    def transcript(self) -> list[Any]:
        return [*self.messages, self.replies[-1]] if self.replies else []

    async def _create(self, **kwargs: Any) -> ChatCompletion:
        response: ChatCompletion = await self._client.chat.completions.create(**kwargs)
        self.messages = list(kwargs['messages'])
        self.replies.append(response.choices[0].message.model_dump(exclude_none=True))
        return response


async def sweep(
    settings: Settings,
    questions: list[EvalQuestion],
    repeats: int,
    runs: list[Run],
    out_path: Path,
    prompts: dict[str, str],
) -> None:
    """Append a Run per completed loop"""
    recorder = RecordingLLM(make_llm_client(settings=settings))
    system_prompt = load_system_prompt(settings)  # the prompt the CLI provides so the numbers describe the real agent

    with out_path.open('w', encoding='utf-8') as out:
        async with open_mcp_session(settings) as mcp_session:
            for entry in questions:
                typer.echo(f'\n{entry.question}')
                for i in range(repeats):
                    recorder.reset()
                    started = time.monotonic()
                    result = await run_agent(
                        question=entry.question,
                        mcp_session=mcp_session,
                        llm_client=cast(AsyncOpenAI, recorder),
                        settings=settings,
                        system_prompt=system_prompt,
                    )
                    seconds = time.monotonic() - started

                    transcript = recorder.transcript()
                    called, failed = tally_tools(transcript)
                    citations, ungrounded = check_citations(result.text, transcript)
                    tool_tokens = count_tool_tokens(transcript)
                    runs.append(
                        Run(
                            question=entry.question,
                            qtype=entry.type,
                            hops=len(recorder.replies),
                            seconds=seconds,
                            tool_tokens=tool_tokens,
                            stopped_reason=result.stopped_reason,
                            called=called,
                            failed=failed,
                            citations=citations,
                            ungrounded=ungrounded,
                            no_coverage=NO_COVERAGE_REPLY in result.text,
                            expected_hit=check_expected(result.text, entry.expected_urls),
                        )
                    )

                    record = {
                        'prompts': prompts,
                        'question': entry.question,
                        'type': entry.type,
                        'repeat': i + 1,
                        'answer': result.text,
                        'stopped_reason': result.stopped_reason,
                        'hops': len(recorder.replies),
                        'seconds': round(seconds, 2),
                        'tool_tokens': tool_tokens,
                        'citations': citations,
                        'ungrounded_urls': ungrounded,
                        'no_coverage': NO_COVERAGE_REPLY in result.text,
                        'expected_urls': entry.expected_urls,
                        'expected_hit': check_expected(result.text, entry.expected_urls),
                        'called': dict(called),
                        'failed': dict(failed),
                        'tool_calls': [{'name': c.name, 'args': c.args} for c in result.tool_calls],
                        'reasoning': [r.get('reasoning') for r in recorder.replies],
                        'transcript': transcript,
                    }
                    out.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                    out.flush()

                    tally = f'{called[SEARCH_TOOL]}s/{failed[SEARCH_TOOL]}f {called[FETCH_TOOL]}f/{failed[FETCH_TOOL]}f'
                    typer.echo(
                        f'  {i + 1}/{repeats}  {result.stopped_reason:<16} '
                        f'{len(recorder.replies)} hops {seconds:5.1f}s  {tally}'
                    )


def _quality_row(label: str, group: list[Run]) -> str:
    """Per run rates: searched maps to retrieved, cited to cited."""
    n = len(group)
    scored = [r for r in group if r.expected_hit is not None]
    cited = f'{sum(1 for r in scored if r.expected_hit) / len(scored):.2f}' if scored else '--'
    return (
        f'| {label} | {n} | {sum(1 for r in group if sum(r.called.values())) / n:.2f} | {cited} | '
        f'{sum(1 for r in group if r.no_coverage) / n:.2f} | {sum(1 for r in group if r.ungrounded) / n:.2f} |'
    )


def _loop_row(label: str, group: list[Run]) -> str:
    return (
        f'| {label} | {len(group)} | {mean(r.hops for r in group):.2f} | {mean(r.seconds for r in group):.1f} | '
        f'{sum(r.called[SEARCH_TOOL] for r in group)} | {sum(r.failed[SEARCH_TOOL] for r in group)} | '
        f'{sum(r.called[FETCH_TOOL] for r in group)} | {sum(r.failed[FETCH_TOOL] for r in group)} |'
    )


def _by_type(runs: list[Run]) -> list[tuple[str, list[Run]]]:
    groups = [(t, [r for r in runs if (r.qtype or 'untyped') == t]) for t in {r.qtype or 'untyped' for r in runs}]
    return sorted(groups, key=lambda g: -len(g[1]))


def report(runs: list[Run], questions: list[EvalQuestion], settings: Settings) -> None:
    total = len(runs)
    if not total:
        typer.echo('no runs completed')
        return

    no_tools = sum(1 for r in runs if not sum(r.called.values()))
    searched = sum(r.called[SEARCH_TOOL] for r in runs)
    search_failed = sum(r.failed[SEARCH_TOOL] for r in runs)
    fetched = sum(r.called[FETCH_TOOL] for r in runs)
    fetch_failed = sum(r.failed[FETCH_TOOL] for r in runs)
    seconds = [r.seconds for r in runs]
    tokens = [float(r.tool_tokens) for r in runs]
    cited = sum(r.citations for r in runs)
    ungrounded = sum(len(r.ungrounded) for r in runs)
    no_coverage = sum(1 for r in runs if r.no_coverage)

    typer.echo(f'\n{"=" * 76}\nruns: {total}  questions: {len(questions)}\n')
    typer.echo(f'hops per run        avg {mean(r.hops for r in runs):>7.2f}')
    typer.echo(
        f'seconds per run     avg {mean(seconds):>7.1f}   p50 {_pct(seconds, 0.5):>6.1f}   '
        f'p95 {_pct(seconds, 0.95):>6.1f}   max {max(seconds):>6.1f}'
    )
    typer.echo(
        f'tool tokens per run avg {mean(tokens):>7.0f}   max {max(tokens):>6.0f}   '
        f'budget {settings.agent_tool_result_token_budget}'
    )
    for tool, called, bad in ((SEARCH_TOOL, searched, search_failed), (FETCH_TOOL, fetched, fetch_failed)):
        typer.echo(f'{tool:<19} Σ called {called:>4}  Σ failed {bad:>4}  {_rate(bad, called)}')
    typer.echo(f'no tool call rate   {no_tools / total:>7.1%}  ({no_tools}/{total})')
    typer.echo(f'no coverage reply   {no_coverage / total:>7.1%}  ({no_coverage}/{total})')
    sourced = _rate(cited - ungrounded, cited)
    typer.echo(f'citations           Σ {cited:>5}   ungrounded {ungrounded}   {sourced} from tool results')

    typer.echo('\nstopped_reason')
    for reason, n in Counter(r.stopped_reason for r in runs).most_common():
        typer.echo(f'  {reason:<18} {n:>4}  {n / total:6.1%}')

    if ungrounded:
        typer.echo('\nURLs cited that no tool returned')
        for url, n in Counter(u for r in runs for u in r.ungrounded).most_common():
            typer.echo(f'  {n:>3}x  {url}')

    typer.echo('\nAnswer quality by type   rates per run, -- when the set carries no expected_urls\n')
    typer.echo('| Type | n | Searched | Cited | Refused | Fabricated |')
    typer.echo('|---|---|---|---|---|---|')
    typer.echo(_quality_row('**Overall**', runs))
    for qtype, group in _by_type(runs):
        typer.echo(_quality_row(f'`{qtype}`', group))

    typer.echo('\nLoop behaviour by type   hops and secs are per run, the rest are totals\n')
    typer.echo('| Type | n | Hops | Secs | Searches | Search fail | Fetches | Fetch fail |')
    typer.echo('|---|---|---|---|---|---|---|---|')
    typer.echo(_loop_row('**Overall**', runs))
    for qtype, group in _by_type(runs):
        typer.echo(_loop_row(f'`{qtype}`', group))

    typer.echo('\nper question')
    typer.echo('| Question | n | Hops | Secs | Searches | Search fail | Fetches | Fetch fail |')
    typer.echo('|---|---|---|---|---|---|---|---|')
    for entry in questions:
        qruns = [r for r in runs if r.question == entry.question]
        if not qruns:
            continue
        label = entry.question if len(entry.question) <= 40 else f'{entry.question[:39]}…'
        typer.echo(_loop_row(label, qruns))


def main(
    questions_file: Annotated[Path, typer.Option(help='JSONL question set', exists=True, readable=True)] = Path(
        '../rag-mcp/eval/queries.jsonl'
    ),
    repeats: Annotated[int, typer.Option(min=1, help='runs per question')] = 3,
    run_dir: Annotated[Path, typer.Option(help='directory to save loop eval run results')] = Path('eval/runs'),
    mcp_tools_file: Annotated[
        Path, typer.Option(help='the server tool prompts, hashed into each record', exists=True, readable=True)
    ] = Path('../rag-mcp/src/rag/prompts/mcp_tools.txt'),
) -> None:
    settings = get_settings()
    try:
        questions = load_questions(questions_file)
    except ValueError as e:
        typer.echo(f'Error loading questions: {e}', err=True)
        raise typer.Exit(1) from e

    system_prompt_file = settings.agent_system_prompt_path or (
        Path(__file__).resolve().parent.parent / 'src/pathfinder_agent/prompts/agent_system.txt'
    )
    prompts = {
        'agent_system': prompt_digest(system_prompt_file),
        'mcp_tools': prompt_digest(mcp_tools_file),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f'{datetime.now():%Y-%m-%dT%H-%M-%S}_loop_eval.jsonl'
    typer.echo(f'{len(questions)} questions x {repeats} against {settings.mcp_server_url} / {settings.llm_model}')
    typer.echo(f'prompts  agent_system {prompts["agent_system"]}  mcp_tools {prompts["mcp_tools"]}')

    runs: list[Run] = []
    try:
        asyncio.run(sweep(settings, questions, repeats, runs, out_path, prompts))
    except KeyboardInterrupt:
        typer.echo('\ninterrupted, reporting what finished', err=True)
    except ExceptionGroup as eg:
        causes = '; '.join(f'{type(e).__name__}: {e}' for e in leaf_causes(eg))
        typer.echo(f'\nMCP server stopped answering ({causes}), reporting what finished', err=True)
    finally:
        report(runs, questions, settings)
        typer.echo(f'\n{len(runs)} runs written to {out_path}')


if __name__ == '__main__':
    typer.run(main)
