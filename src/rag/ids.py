from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def make_document_id(content: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Return a stable ID for a chunk.

    The ID is derived from the source path, chunk index, and content. Re-ingesting
    an unchanged document therefore targets the same rows instead of creating
    duplicates, while an edited chunk gets a new ID.
    """
    meta = metadata or {}
    source = str(meta.get("source", ""))
    chunk_index = str(meta.get("chunk_index", ""))
    raw = f"{source}\x00{chunk_index}\x00{content}"
    return f"doc_{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def dedupe_by_id(
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Drop repeated IDs within a batch, keeping the last occurrence.

    ChromaDB rejects a batch that contains the same ID twice. Duplicate IDs can
    happen when a source file is passed twice in one ingestion call.
    """
    last: dict[str, int] = {}
    for i, doc_id in enumerate(ids):
        last[doc_id] = i

    unique_ids: list[str] = []
    unique_docs: list[str] = []
    unique_meta: list[dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        if last[doc_id] != i:
            continue
        unique_ids.append(doc_id)
        unique_docs.append(documents[i])
        unique_meta.append(metadatas[i])
    return unique_ids, unique_docs, unique_meta
