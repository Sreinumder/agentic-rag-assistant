from __future__ import annotations

import chromadb

from src.config import Config
from src.rag.document_loader import Document
from src.rag.embeddings import get_embeddings, get_query_embedding


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name="assistant_docs",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.content for doc in documents]
        ids = [f"doc_{i}_{hash(doc.content) & 0xFFFFFFFF:08x}" for i, doc in enumerate(documents)]
        metadatas = [doc.metadata for doc in documents]

        embeddings = get_embeddings(texts)

        batch_size = 100
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

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
