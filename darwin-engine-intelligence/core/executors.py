"""
Executors for evaluating individuals either locally or via a distributed backend.
"""
from __future__ import annotations
from typing import Callable, Iterable, List, Any


class LocalExecutor:
    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
        return [fn(x) for x in items]


class RayExecutor:
    def __init__(self) -> None:
        self._ok = False
        try:
            import ray  # noqa: F401
            self._ok = True
        except Exception:
            self._ok = False

    def available(self) -> bool:
        return self._ok

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
        if not self._ok:
            # Fallback local
            return [fn(x) for x in items]
        import ray
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        @ray.remote
        def _remote_call(v):
            return fn(v)

        futures = [_remote_call.remote(x) for x in items]
        results = ray.get(futures)
        return list(results)
