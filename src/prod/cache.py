from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class TTLCache:
    """Simple thread-safe TTL cache for prompt/response caching."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._data: dict[str, tuple[float, Any]] = {}

    def _key(self, prompt: str, context: str, temperature: float) -> str:
        raw = json.dumps({"p": prompt, "c": context, "t": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, context: str, temperature: float) -> Any | None:
        key = self._key(prompt, context, temperature)
        item = self._data.get(key)
        if item is None:
            return None
        timestamp, value = item
        if time.monotonic() - timestamp > self._ttl:
            del self._data[key]
            return None
        return value

    def set(self, prompt: str, context: str, temperature: float, value: Any) -> None:
        if len(self._data) >= self._max_size:
            self._evict()
        key = self._key(prompt, context, temperature)
        self._data[key] = (time.monotonic(), value)

    def _evict(self) -> None:
        # Remove oldest entries to keep cache bounded
        oldest_keys = sorted(self._data, key=lambda k: self._data[k][0])
        for k in oldest_keys[: self._max_size // 4]:
            del self._data[k]

    def clear(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)
