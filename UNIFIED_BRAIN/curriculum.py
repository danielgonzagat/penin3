"""
Curriculum + replay harness for long runs.
- Fixed rules, multi-seed evaluation gates, replay of best configs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ReplayEntry:
    config: Dict[str, int | float]
    seed: int
    scores: Dict[str, float]

@dataclass
class CurriculumHarness:
    tasks: List[str] = field(default_factory=lambda: [
        'cartpole_nominal', 'cartpole_noise', 'cartpole_shift', 'mountaincar', 'acrobot'
    ])
    replay: List[ReplayEntry] = field(default_factory=list)

    def push_if_better(self, config: Dict[str, int | float], seed: int, scores: Dict[str, float]) -> None:
        # Keep only configs that beat median of replay
        try:
            med = self._median_best()
            total = float(sum(scores.values()) / max(1, len(scores)))
            if total > med:
                self.replay.append(ReplayEntry(config=config, seed=seed, scores=scores))
                self.replay = self.replay[-200:]  # cap
        except Exception:
            pass

    def _median_best(self) -> float:
        if not self.replay:
            return 0.0
        vals = [float(sum(e.scores.values()) / max(1, len(e.scores))) for e in self.replay]
        vals.sort()
        return vals[len(vals)//2]
