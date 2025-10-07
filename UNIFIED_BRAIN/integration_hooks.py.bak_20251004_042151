#!/usr/bin/env python3
"""
Phase 1 Integration Hooks for Unified Brain
- Gödel Incompleteness monitor (anti-stagnation)
- Needle-based meta-controller (safe, lightweight adapter)
- Composite model builder to expose trainable parameters for monitoring
"""
from __future__ import annotations

import os
import math
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

# Optional import for Gödel incompleteness engine
try:
    from intelligence_system.extracted_algorithms.incompleteness_engine import (
        EvolvedGodelianIncompleteness,
    )
    _GODEL_AVAILABLE = True
except Exception:
    EvolvedGodelianIncompleteness = None  # type: ignore
    _GODEL_AVAILABLE = False


def build_trainable_composite(controller: Any, registry: Any, router: Any) -> nn.Module:
    """Build a composite nn.Module aggregating trainable submodules/parameters.

    This module is NOT added to the optimizer. It exists solely to expose parameters
    to monitoring utilities (e.g., Gödel engine).
    """
    class TrainableComposite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules: List[nn.Module] = []

            # Top active neurons adapters (if available)
            try:
                active_neurons = registry.get_active()
                # Limit to first 16 for V3 fast path
                for neuron in active_neurons[:16]:
                    if hasattr(neuron, "A_in") and isinstance(neuron.A_in, nn.Module):
                        modules.append(neuron.A_in)
                    if hasattr(neuron, "A_out") and isinstance(neuron.A_out, nn.Module):
                        modules.append(neuron.A_out)
            except Exception:
                pass

            # V7 bridge (policy/value head)
            try:
                if hasattr(controller, "v7_bridge") and isinstance(controller.v7_bridge, nn.Module):
                    modules.append(controller.v7_bridge)
            except Exception:
                pass

            # Aggregate as ModuleList so named_parameters() traverses correctly
            self.modules_list = nn.ModuleList(modules)

            # Router competence parameter (if present)
            try:
                if router is not None and hasattr(router, "competence") and isinstance(router.competence, torch.nn.Parameter):
                    # Register parameter reference for monitoring
                    self.router_competence = router.competence  # type: ignore[attr-defined]
                else:
                    # Dummy parameter for shape consistency
                    self.router_competence = nn.Parameter(torch.zeros(()), requires_grad=False)
            except Exception:
                self.router_competence = nn.Parameter(torch.zeros(()), requires_grad=False)

    return TrainableComposite()


@dataclass
class GodelResult:
    is_stagnant: bool
    stagnation_score: float
    actions: List[str]
    signals: Dict[str, float]


class GodelMonitor:
    """Safe wrapper around EvolvedGodelianIncompleteness to apply anti-stagnation."""

    def __init__(self, delta_0: float = 0.05) -> None:
        self.enabled = _GODEL_AVAILABLE and os.getenv("ENABLE_GODEL", "1") == "1"
        self._engine = EvolvedGodelianIncompleteness(delta_0=delta_0) if self.enabled else None

    def apply(
        self,
        composite_model: nn.Module,
        optimizer: optim.Optimizer,
        loss_value: float,
        accuracy: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> Optional[GodelResult]:
        if not self.enabled or self._engine is None:
            return None

        result = self._engine.apply_incompleteness_evolved(
            model=composite_model,
            optimizer=optimizer,
            loss=loss_value,
            accuracy=accuracy,
            batch_size=batch_size,
        )
        return GodelResult(
            is_stagnant=bool(result.get("is_stagnant", False)),
            stagnation_score=float(result.get("stagnation_score", 0.0)),
            actions=list(result.get("actions", [])),
            signals=dict(result.get("signals", {})),
        )


class NeedleMetaController:
    """Lightweight meta-controller adapter.

    Attempts to use THE_NEEDLE's MetaLearner if available; otherwise applies a
    conservative heuristic to adjust LR and dropout based on recent stats.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_NEEDLE_META", "1") == "1"
        self._needle_meta = None
        self._prev_loss: Optional[float] = None
        self._cooldown = 0

        if self.enabled:
            # Best-effort import from THE_NEEDLE.py, may be heavy; keep optional
            try:
                import importlib.util
                needle_path = "/root/THE_NEEDLE.py"
                if os.path.exists(needle_path):
                    spec = importlib.util.spec_from_file_location("THE_NEEDLE", needle_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore
                        # Prefer a class named MetaLearner if it exists
                        self._needle_meta = getattr(module, "MetaLearner", None)
            except Exception:
                self._needle_meta = None

    def adjust_optimizer(
        self,
        optimizer: optim.Optimizer,
        stats: Dict[str, Any],
        episode_reward: float,
        loss_value: float,
    ) -> None:
        if not self.enabled:
            return

        # Simple, safe heuristic when THE_NEEDLE meta is not available
        try:
            # Cooldown to avoid aggressive thrashing
            if self._cooldown > 0:
                self._cooldown -= 1
                self._prev_loss = loss_value
                return

            # Pull a scalar LR (first param group)
            current_lr = optimizer.param_groups[0].get("lr", 3e-4)

            # If loss is rising and reward not improving, reduce LR slightly
            loss_rising = self._prev_loss is not None and loss_value > (self._prev_loss * 1.02)
            recent_avg = float(stats.get("avg_reward_last_100", 0.0))
            best = float(stats.get("best_reward", 0.0))
            reward_stagnant = recent_avg < max(1.0, best * 0.5)  # very loose

            new_lr = current_lr
            if loss_rising and reward_stagnant:
                new_lr = max(current_lr * 0.8, 1e-6)
                self._cooldown = 5
            else:
                # Small exploration up to +10% when stable
                new_lr = min(current_lr * 1.05, 1e-2)

            optimizer.param_groups[0]["lr"] = float(new_lr)
            self._prev_loss = loss_value
        except Exception:
            # Never break training on meta-control failure
            self._prev_loss = loss_value
            return
