from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import Config
from src.rag.embeddings import get_embeddings, get_query_embedding
from src.rag.ids import dedupe_by_id, make_document_id

if TYPE_CHECKING:
    from src.rag.document_loader import Document

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        # Imported lazily so the rest of the package (tools, agent, tests) can be
        # used without the heavy ChromaDB dependency installed.
        import chromadb

        self._client = chromadb.PersistentClient(
            path=persist_dir or Config.CHROMA_PERSIST_DIR
        )
        self._collection = self._client.get_or_create_collection(
            name="assistant_docs",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(self, documents: list[Document]) -> int:
        """Embed and upsert chunks.

        IDs are content-derived, so ingesting the same corpus twice updates rows
        instead of duplicating them.
        """
        if not documents:
            return 0

        texts = [doc.content for doc in documents]
        metadatas = [dict(doc.metadata) for doc in documents]
        ids = [make_document_id(text, meta) for text, meta in zip(texts, metadatas)]

        ids, texts, metadatas = dedupe_by_id(ids, texts, metadatas)
        if not texts:
            return 0

        embeddings = get_embeddings(texts)

        for start in range(0, len(texts), BATCH_SIZE):
            end = start + BATCH_SIZE
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Upserted %d chunks", len(texts))
        return len(texts)

    def query(self, query_text: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or Config.TOP_K
        if self._collection.count() == 0:
            return []

        query_embedding = get_query_embedding(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
            )
        return output
