from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe TTL cache for prompt/response pairs.

    FastAPI serves synchronous handlers from a thread pool, so every read and
    write is guarded by a re-entrant lock. Eviction removes the oldest quarter
    when the cache reaches its size limit.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max(1, max_size)
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(prompt: str, context: str, temperature: float) -> str:
        raw = json.dumps({"p": prompt, "c": context, "t": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, context: str, temperature: float) -> Any | None:
        key = self._key(prompt, context, temperature)
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            timestamp, value = item
            if time.monotonic() - timestamp > self._ttl:
                del self._data[key]
                return None
            return value

    def set(self, prompt: str, context: str, temperature: float, value: Any) -> None:
        key = self._key(prompt, context, temperature)
        with self._lock:
            if len(self._data) >= self._max_size:
                self._evict()
            self._data[key] = (time.monotonic(), value)

    def _evict(self) -> None:
        # Called with the lock held. Drop the oldest entries to keep the cache
        # bounded without clearing everything on each miss.
        oldest_keys = sorted(self._data, key=lambda k: self._data[k][0])
        for key in oldest_keys[: max(1, self._max_size // 4)]:
            del self._data[key]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)
