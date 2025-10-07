"""
PENIN-Ω ACFA League
====================

Shadow/Canary deployment orchestration with automatic rollback.

Implements champion-challenger system with:
- L∞ non-compensatory metric aggregation
- Ethics gates (ΣEA/LO-14)
- Shadow/Canary deployment
- Automatic rollback on violations
"""

from __future__ import annotations

from penin.league.acfa_league import (
    ACFALeague,
    DeploymentStage,
    LeagueConfig,
    ModelCandidate,
    ModelMetrics,
    PromotionDecision,
)

__all__ = [
    "ACFALeague",
    "ModelCandidate",
    "ModelMetrics",
    "DeploymentStage",
    "PromotionDecision",
    "LeagueConfig",
]
