import hashlib
import logging

import pytest

from rag import corpus
from rag.config import Settings
from rag.corpus import chunk_corpus
from rag.models import Article, Chunk, ChunksManifest


def _make_article(doc_id: str = 'bestiary__x__y') -> Article:
    return Article(
        doc_id=doc_id,
        url='https://www.d20pfsrd.com/bestiary/x/y',
        title='XY',
        category='bestiary',
        breadcrumb=['Home'],
        body_md='body',
        n_chars=4,
    )


def _make_chunk(n_tokens: int, chunk_id: str = 'bestiary__x__y#000') -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id='bestiary__x__y',
        heading_path=[],
        text='body',
        category='bestiary',
        n_tokens=n_tokens,
    )


def _patch_chunk_article(monkeypatch: pytest.MonkeyPatch, n_tokens: int) -> None:
    monkeypatch.setattr(
        corpus,
        'chunk_article',
        lambda *_args, **_kwargs: [_make_chunk(n_tokens)],
    )


def test_chunk_corpus_empty_articles_returns_empty():
    assert chunk_corpus([], tokenizer=None, max_tokens=450, overlap=50) == []


def test_chunk_corpus_raises_when_chunk_exceeds_hard_limit(monkeypatch: pytest.MonkeyPatch):
    # 450 * 1.02 -> hard_limit 459. 500 is past the BPE slack and must fail
    _patch_chunk_article(monkeypatch, n_tokens=500)
    with pytest.raises(ValueError, match='beyond BPE slack'):
        chunk_corpus([_make_article()], tokenizer=None, max_tokens=450, overlap=50)


def test_chunk_corpus_warns_on_drift_within_slack(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    # 451 is over max_tokens but under the 459 hard_limit -> warning, no  raise
    _patch_chunk_article(monkeypatch, n_tokens=451)
    with caplog.at_level(logging.WARNING):
        chunks = chunk_corpus([_make_article()], tokenizer=None, max_tokens=450, overlap=50)
    assert len(chunks) == 1
    assert 'within BPE slack' in caplog.text


def test_chunk_corpus_within_budget_does_not_warn(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _patch_chunk_article(monkeypatch, n_tokens=440)
    with caplog.at_level(logging.WARNING):
        chunks = chunk_corpus([_make_article('a'), _make_article('b')], tokenizer=None, max_tokens=450, overlap=50)
    assert len(chunks) == 2
    assert caplog.text == ''


def test_chunks_manifest_build_records_params_and_source_hash(tmp_path):
    source = tmp_path / 'corpus.parquet'
    source.write_bytes(b'parquet-bytes')
    settings = Settings()

    manifest = ChunksManifest.build(settings, source, n_articles=12, n_chunks=34)

    assert manifest.source_file == str(source)
    assert manifest.source_sha256 == hashlib.sha256(b'parquet-bytes').hexdigest()
    assert manifest.n_articles == 12
    assert manifest.n_chunks == 34
    assert manifest.min_body_length == settings.min_body_length
    assert manifest.tokenizer_model == settings.tokenizer_model
    assert manifest.max_tokens == settings.chunk_max_tokens
    assert manifest.overlap == settings.chunk_overlap
    assert manifest.created_at.tzinfo is not None
