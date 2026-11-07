from __future__ import annotations

import threading

import pytest

from src.prod import cache as cache_module
from src.prod.cache import TTLCache


def test_set_then_get() -> None:
    cache = TTLCache()
    cache.set("prompt", "context", 0.3, {"text": "answer"})
    assert cache.get("prompt", "context", 0.3) == {"text": "answer"}


def test_miss_returns_none() -> None:
    cache = TTLCache()
    assert cache.get("missing", "", 0.3) is None


def test_key_includes_context_and_temperature() -> None:
    cache = TTLCache()
    cache.set("prompt", "context-a", 0.3, "a")
    assert cache.get("prompt", "context-b", 0.3) is None
    assert cache.get("prompt", "context-a", 0.9) is None


def test_expired_entry_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock["now"])

    cache = TTLCache(ttl_seconds=10)
    cache.set("prompt", "", 0.3, "answer")

    clock["now"] = 105.0
    assert cache.get("prompt", "", 0.3) == "answer"

    clock["now"] = 111.0
    assert cache.get("prompt", "", 0.3) is None
    assert cache.size == 0


def test_cache_never_exceeds_max_size() -> None:
    cache = TTLCache(max_size=10)
    for i in range(100):
        cache.set(f"prompt-{i}", "", 0.3, i)
    assert cache.size <= 10


def test_eviction_drops_oldest_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock["now"])

    cache = TTLCache(max_size=4)
    for i in range(4):
        clock["now"] = float(i)
        cache.set(f"p{i}", "", 0.3, i)

    clock["now"] = 4.0
    cache.set("p4", "", 0.3, 4)

    assert cache.size == 4
    assert cache.get("p0", "", 0.3) is None
    assert cache.get("p4", "", 0.3) == 4


def test_clear() -> None:
    cache = TTLCache()
    cache.set("p", "", 0.3, 1)
    cache.clear()
    assert cache.size == 0


def test_concurrent_access_is_safe() -> None:
    cache = TTLCache(max_size=50)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(200):
                cache.set(f"p-{worker_id}-{i}", "", 0.3, i)
                cache.get(f"p-{worker_id}-{i}", "", 0.3)
                cache.size
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cache.size <= 50
