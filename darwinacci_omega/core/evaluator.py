from __future__ import annotations

import os, random
from typing import Callable, Dict, Any, List, Optional


Genome = Dict[str, float]
EvalFn = Callable[[Genome, random.Random], Dict[str, Any]]


class EvaluatorPipeline:
    """
    Wraps a base eval_fn and augments metrics with optional intrinsic signals:
    - RND-like curiosity over genome/behavior projections (cheap, dependency-free)
    - World-model stub (predictive error placeholder if no obs available)
    - Portfolio of tasks (routes to sub-evaluators)
    Controlled by env vars; defaults keep base behavior if disabled.
    """

    def __init__(self, base: EvalFn, portfolio: Optional[List[EvalFn]] = None, task_names: Optional[List[str]] = None):
        self.base = base
        self.portfolio = portfolio or []
        self.task_names = task_names or [f"task_{i}" for i in range(len(self.portfolio))]
        # Curiosity knobs
        self.use_rnd = os.getenv('DARWINACCI_RND', '0') == '1'
        self.rnd_dim = int(os.getenv('DARWINACCI_RND_DIM', '8'))
        # World model stub toggle
        self.use_world = os.getenv('DARWINACCI_WORLD', '0') == '1'
        # Portfolio toggle
        self.use_portfolio = os.getenv('DARWINACCI_PORTFOLIO', '0') == '1' and len(self.portfolio) > 0
        # Curriculum (epsilon-greedy over recent means)
        self.use_curriculum = os.getenv('DARWINACCI_CURRICULUM', '0') == '1' and self.use_portfolio
        self.epsilon = float(os.getenv('DARWINACCI_CURR_EPS', '0.2'))
        self._stats: Dict[str, List[float]] = {n: [] for n in self.task_names}

    def _rnd_curiosity(self, genome: Genome, behavior: List[float], rng: random.Random) -> float:
        # Cheap hashing-based projection of genome+behavior to emulate error signal
        keys = sorted(genome.keys())[: self.rnd_dim]
        vals = [float(genome.get(k, 0.0)) for k in keys]
        vec = vals + list(behavior[: max(0, self.rnd_dim - len(vals))])
        h = 0.0
        for v in vec:
            h = (h * 1315423911.0 + float(v)) % 1_000_000.0
        # Normalize to [0,1]
        return (h % 1000.0) / 1000.0

    def _world_stub(self, genome: Genome, rng: random.Random) -> float:
        # Placeholder predictive error proxy based on genome entropy/spread
        if not genome:
            return 0.0
        vals = [abs(float(v)) for v in genome.values()]
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        # Normalize
        return min(1.0, (mean + var) / (1.0 + mean + var))

    def _choose_task_index(self, rng: random.Random) -> int:
        if not self.use_curriculum:
            return rng.randint(0, len(self.portfolio) - 1)
        # epsilon-greedy on avg objective
        if rng.random() < self.epsilon:
            return rng.randint(0, len(self.portfolio) - 1)
        # exploit best average
        best_idx = 0
        best_val = float('-inf')
        for i, name in enumerate(self.task_names):
            vals = self._stats.get(name) or []
            avg = sum(vals) / len(vals) if vals else 0.0
            if avg > best_val:
                best_val = avg
                best_idx = i
        return best_idx

    def evaluate(self, genome: Genome, rng: random.Random) -> Dict[str, Any]:
        # Route to a sub-task if portfolio enabled; else base
        selected_task: Optional[str] = None
        if self.use_portfolio:
            idx = self._choose_task_index(rng)
            selected_task = self.task_names[idx] if idx < len(self.task_names) else None
            out = dict(self.portfolio[idx](genome, rng))
        else:
            out = dict(self.base(genome, rng))

        behavior = out.get('behavior') or [float(genome.get('hidden_size', 64)) / 256.0,
                                           float(genome.get('learning_rate', 1e-3)) * 1000.0]

        if self.use_rnd:
            rnd = self._rnd_curiosity(genome, behavior, rng)
            out['curiosity_rnd'] = float(rnd)
            # Blend into objective as small bonus if not provided externally
            if 'objective' in out:
                out['objective'] = float(out['objective']) * 0.95 + rnd * 0.05

        if self.use_world:
            wm = self._world_stub(genome, rng)
            out['world_error'] = float(wm)
            if 'objective' in out:
                out['objective'] = float(out['objective']) * 0.97 + wm * 0.03

        # Update curriculum stats
        try:
            if self.use_portfolio and selected_task is not None and 'objective' in out:
                self._stats.setdefault(selected_task, []).append(float(out['objective']))
                # cap memory per task
                if len(self._stats[selected_task]) > 200:
                    self._stats[selected_task] = self._stats[selected_task][-200:]
        except Exception:
            pass

        return out
