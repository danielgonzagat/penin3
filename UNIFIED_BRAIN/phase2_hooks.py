#!/usr/bin/env python3
"""
Phase 2 Hooks
- SelfObserver: tracks entropy/step-time signals and computes intrinsic reward
- MultiObjectiveScheduler: adapts intrinsic weight and core.top_k per episode
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import time

import torch


class SelfObserver:
    """Collects per-step signals and computes intrinsic reward shaping."""

    def __init__(self) -> None:
        self.step_count: int = 0
        self.total_intrinsic: float = 0.0
        self.prev_entropy: Optional[float] = None
        self.curiosity_ema: float = 0.0
        self.step_time_ema: float = 0.0
        self.ema_beta: float = 0.9

    def record_step(self, action_probs: torch.Tensor, step_time: float) -> float:
        # Categorical entropy H(p) = -Σ p log p
        p = torch.clamp(action_probs, 1e-8, 1.0)
        entropy = float(-(p * p.log()).sum().item())

        # Surprise as absolute entropy change
        surprise = 0.0
        if self.prev_entropy is not None:
            surprise = abs(entropy - self.prev_entropy)
        self.prev_entropy = entropy

        # Update EMAs
        self.curiosity_ema = self.ema_beta * self.curiosity_ema + (1 - self.ema_beta) * entropy
        self.step_time_ema = self.ema_beta * self.step_time_ema + (1 - self.ema_beta) * float(step_time)

        # Intrinsic signal combines entropy and surprise
        intrinsic_signal = entropy * 0.7 + surprise * 0.3
        self.step_count += 1
        self.total_intrinsic += intrinsic_signal
        return intrinsic_signal

    def episode_summary(self) -> Dict[str, float]:
        avg_intrinsic = (self.total_intrinsic / max(self.step_count, 1))
        return {
            'avg_intrinsic': avg_intrinsic,
            'curiosity_ema': self.curiosity_ema,
            'step_time_ema': self.step_time_ema,
        }


@dataclass
class SchedulerDecision:
    intrinsic_weight: float
    top_k: int
    objective: str


class MultiObjectiveScheduler:
    """Adapts intrinsic weight and top_k to balance speed/learning/exploration."""

    def __init__(self, core: Any, initial_weight: float = 0.05) -> None:
        self.core = core
        self.intrinsic_weight: float = initial_weight
        self.objective: str = 'accuracy'
        # Bounds
        self.min_weight = 0.03
        self.max_weight = 0.25
        self.min_topk = 4
        self.max_topk = 16

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def decide(self, stats: Dict[str, Any], obs: SelfObserver) -> SchedulerDecision:
        avg_time = float(stats.get('avg_time_per_step', 0.0) or 0.0)
        progress = float(stats.get('learning_progress', 0.0) or 0.0)
        best = float(stats.get('best_reward', 0.0) or 0.0)
        avg100 = float(stats.get('avg_reward_last_100', 0.0) or 0.0)
        curiosity = obs.curiosity_ema

        # Choose objective heuristically
        if avg_time > 0.8:  # P1.4 fix  # slow → prioritize speed
            self.objective = 'speed'
        elif progress < 0.02 and curiosity < 0.5:  # P1.4 fix  # not learning + low curiosity → explore
            self.objective = 'exploration'
        elif avg100 > 0 and best > 0 and avg100 > 0.85 * best:  # P1.4 fix  # near best → robustness
            self.objective = 'robustness'
        else:
            self.objective = 'accuracy'

        # Adjust intrinsic weight
        if self.objective == 'exploration':
            self.intrinsic_weight = self._clamp(self.intrinsic_weight * 1.2 + 0.01, self.min_weight, self.max_weight)
        elif self.objective == 'speed':
            self.intrinsic_weight = self._clamp(self.intrinsic_weight * 0.9, self.min_weight, self.max_weight)
        elif self.objective == 'robustness':
            self.intrinsic_weight = self._clamp(self.intrinsic_weight * 0.95, self.min_weight, self.max_weight)
        else:  # accuracy
            self.intrinsic_weight = self._clamp(self.intrinsic_weight, self.min_weight, self.max_weight)

        # Adjust top_k in core
        current_topk = getattr(self.core, 'top_k', 8) or 8
        new_topk = current_topk
        if self.objective == 'speed':
            new_topk = max(self.min_topk, current_topk // 2)
        elif self.objective == 'exploration':
            new_topk = min(self.max_topk, current_topk + 1)  # P1.4 fix
        elif self.objective == 'robustness':
            new_topk = max(self.min_topk, current_topk - 1)
        # accuracy → keep

        try:
            self.core.top_k = int(new_topk)
        except Exception:
            pass

        return SchedulerDecision(
            intrinsic_weight=self.intrinsic_weight,
            top_k=int(new_topk),
            objective=self.objective,
        )
