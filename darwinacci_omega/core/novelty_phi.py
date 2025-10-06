import os, math
from typing import List, Optional


class Novelty:
    """
    K-NN novelty with optional FAISS backend. Supports N-D behavior.
    Configure with DARWINACCI_NOVELTY_BACKEND in {"faiss","naive"} (default naive).
    """

    def __init__(self, k: int = 7, max_size: int = 1500):
        self.k = k
        self.max_size = max_size
        self.mem: List[List[float]] = []
        # Backend selection
        self._backend = (os.getenv('DARWINACCI_NOVELTY_BACKEND', 'naive') or 'naive').lower()
        self._faiss = None
        if self._backend == 'faiss':
            try:
                import faiss  # type: ignore
                self._faiss = faiss
            except Exception:
                self._backend = 'naive'

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))

    def _score_naive(self, b: List[float]) -> float:
        if not self.mem:
            return 0.0
        d = sorted(self._dist(b, x) for x in self.mem)
        k = min(self.k, len(d))
        return sum(d[:k]) / max(1, k)

    def _score_faiss(self, b: List[float]) -> float:
        if not self.mem or self._faiss is None:
            return 0.0
        import numpy as np  # type: ignore
        # Build index on the fly (small mem sizes) for simplicity
        dim = max(len(v) for v in self.mem + [b])
        def pad(v):
            if len(v) == dim:
                return v
            return v + [0.0] * (dim - len(v))
        xb = np.array([pad(v) for v in self.mem], dtype='float32')
        xq = np.array([pad(b)], dtype='float32')
        index = self._faiss.IndexFlatL2(dim)
        index.add(xb)
        k = min(self.k, len(self.mem))
        D, _ = index.search(xq, k)
        return float(D[0].mean()) if k > 0 else 0.0

    def score(self, b: List[float]) -> float:
        if self._backend == 'faiss':
            return self._score_faiss(b)
        return self._score_naive(b)

    def add(self, b: List[float]):
        if b:
            self.mem.append(list(b))
            if len(self.mem) > self.max_size:
                self.mem.pop(0)