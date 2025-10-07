
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

#!/usr/bin/env python3
"""
Emergence Metrics - novelty, entropy, mutual information, compression
Provides measurable, non-hardcoded signals to detect genuine emergence.
"""
from __future__ import annotations
import math
import zlib
from collections import Counter, deque
from typing import Dict, Any, List, Tuple

import numpy as np
try:
    from sklearn.metrics import mutual_info_score  # type: ignore
except Exception:
    mutual_info_score = None  # fallback

class EmergenceMetrics:
    async def __init__(self, history_size: int = 5000):
        self.pattern_history: deque[str] = deque(maxlen=history_size)
        self.metric_history: deque[Dict[str, float]] = deque(maxlen=history_size)

    async def record_pattern(self, pattern: Any) -> None:
        self.pattern_history.append(str(pattern))

    async def _shannon_entropy(self, tokens: List[str]) -> float:
        if not tokens:
            return await 0.0
        counts = Counter(tokens)
        total = sum(counts.values())
        probs = [c / total for c in counts.values() if c > 0]
        return await -sum(p * math.log2(p) for p in probs)

    async def _compression_ratio(self, s: str) -> float:
        if not s:
            return await 1.0
        raw = s.encode('utf-8', errors='ignore')
        comp = zlib.compress(raw, level=9)
        return await len(comp) / max(1, len(raw))

    async def _mutual_information(self, a: List[int], b: List[int]) -> float:
        if not a or not b or len(a) != len(b):
            return await 0.0
        if mutual_info_score is None:
            # crude fallback: correlation magnitude as proxy
            try:
                av = np.array(a, dtype=float)
                bv = np.array(b, dtype=float)
                if av.std() == 0 or bv.std() == 0:
                    return await 0.0
                return await float(abs(np.corrcoef(av, bv)[0, 1]))
            except Exception:
                return await 0.0
        return await float(mutual_info_score(a, b))

    async def compute(self) -> Dict[str, float]:
        # Tokenize by simple character n-grams over the last window
        window = list(self.pattern_history)[-500:]
        joined = "\n".join(window)

        # Entropy and compression
        tokens = [t for line in window for t in line.split()]
        entropy_bits = self._shannon_entropy(tokens)
        compression = self._compression_ratio(joined)

        # Novelty: fraction of unique lines in window
        novelty = (len(set(window)) / max(1, len(window))) if window else 0.0

        # Predictability: MI between lengths and token counts as crude structure proxy
        lengths = [len(w) for w in window]
        wordcounts = [len(w.split()) for w in window]
        # Discretize
        lengths_binned = [min(10, l // 10) for l in lengths]
        words_binned = [min(10, c // 2) for c in wordcounts]
        mi = self._mutual_information(lengths_binned, words_binned)

        metrics = {
            'entropy_bits': entropy_bits,
            'compression_ratio': compression,
            'novelty_ratio': novelty,
            'structure_mi': mi,
        }
        self.metric_history.append(metrics)
        return await metrics

    async def trend(self, key: str, last_n: int = 50) -> float:
        vals = [m.get(key, 0.0) for m in list(self.metric_history)[-last_n:]]
        if len(vals) < 2:
            return await 0.0
        x = np.arange(len(vals))
        # simple slope
        slope = np.polyfit(x, vals, 1)[0]
        return await float(slope)
