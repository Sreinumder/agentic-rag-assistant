from __future__ import annotations

from src.rag.ids import dedupe_by_id, make_document_id


def test_id_is_stable_across_calls() -> None:
    meta = {"source": "docs/a.md", "chunk_index": 0}
    assert make_document_id("hello world", meta) == make_document_id("hello world", meta)


def test_id_changes_with_content_source_or_index() -> None:
    base = {"source": "docs/a.md", "chunk_index": 0}
    base_id = make_document_id("hello", base)

    assert make_document_id("hello!", base) != base_id
    assert make_document_id("hello", {**base, "source": "docs/b.md"}) != base_id
    assert make_document_id("hello", {**base, "chunk_index": 1}) != base_id


def test_id_is_prefixed_sha256() -> None:
    doc_id = make_document_id("hello", {"source": "a", "chunk_index": 0})
    assert doc_id.startswith("doc_")
    assert len(doc_id) == len("doc_") + 64
    int(doc_id.removeprefix("doc_"), 16)


def test_id_handles_missing_metadata() -> None:
    assert make_document_id("hello") == make_document_id("hello", {})


def test_dedupe_keeps_last_occurrence() -> None:
    ids = ["a", "b", "a"]
    docs = ["first", "bee", "second"]
    metas = [{"n": 1}, {"n": 2}, {"n": 3}]

    unique_ids, unique_docs, unique_meta = dedupe_by_id(ids, docs, metas)

    assert unique_ids == ["b", "a"]
    assert unique_docs == ["bee", "second"]
    assert unique_meta == [{"n": 2}, {"n": 3}]


def test_dedupe_noop_for_unique_ids() -> None:
    ids = ["a", "b"]
    docs = ["one", "two"]
    metas = [{}, {}]
    assert dedupe_by_id(ids, docs, metas) == (ids, docs, metas)
