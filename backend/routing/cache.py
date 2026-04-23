from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class SimpleLRUCache(Generic[K, V]):
    """Tiny in-process LRU cache for routing primitives."""

    def __init__(self, max_size: int = 20000):
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        value = self._store.get(key)
        if value is None:
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
