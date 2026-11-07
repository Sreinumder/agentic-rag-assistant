from __future__ import annotations

from google import genai

from src.config import Config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


def get_embeddings(documents: list[str]) -> list[list[float]]:
    client = _get_client()
    result = client.models.embed_content(
        model=Config.GEMINI_EMBEDDING_MODEL,
        contents=documents,
    )
    return [e.values for e in result.embeddings]


def get_query_embedding(query: str) -> list[float]:
    return get_embeddings([query])[0]
