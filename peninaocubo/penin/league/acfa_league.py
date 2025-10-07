"""
PENIN-Ω ACFA League — Autonomous Champion-Challenger Feedback Adaptation
==========================================================================

Implements shadow/canary deployment orchestration with automatic rollback
based on L∞ non-compensatory aggregation and ΣEA/LO-14 ethical gates.

References:
- PENIN_OMEGA_COMPLETE_EQUATIONS_GUIDE.md § 7 (ACFA EPV equation)
- README.md § Features (ACFA League)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.10 compatibility
UTC = timezone.utc


# ============================================================================
# Types & Enums
# ============================================================================


class DeploymentStage(str, Enum):
    """Deployment stages in the champion-challenger pipeline."""

    SHADOW = "shadow"  # Model runs in shadow mode (metrics only, no traffic)
    CANARY = "canary"  # Model serves % of traffic (canary deployment)
    PRODUCTION = "production"  # Model is the current champion


class PromotionDecision(str, Enum):
    """Decision outcome for challenger promotion."""

    PROMOTED = "promoted"  # Challenger becomes new champion
    REJECTED = "rejected"  # Challenger rejected, keep current champion
    ROLLBACK = "rollback"  # Active rollback due to ethics/performance failure


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ModelMetrics:
    """
    Model performance metrics for champion-challenger evaluation.

    Uses L∞ non-compensatory aggregation: worst dimension dominates.
    """

    accuracy: float = 0.0
    robustness: float = 0.0
    calibration: float = 0.0
    fairness: float = 0.0
    privacy: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for L∞ computation."""
        return {
            "accuracy": self.accuracy,
            "robustness": self.robustness,
            "calibration": self.calibration,
            "fairness": self.fairness,
            "privacy": self.privacy,
        }

    def linf_score(self) -> float:
        """
        Compute L∞ score (non-compensatory, bottleneck = min dimension).

        Rationale:
        - Non-compensatory ethics require the worst dimension to dominate
        - Using min() ensures no strong dimension compensates for a weak one
        """
        values = [
            self.accuracy,
            self.robustness,
            self.calibration,
            self.fairness,
            self.privacy,
        ]

        # Filter invalid/zero values (treat as 0.0)
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return 0.0

        return float(min(valid_values))


@dataclass
class ModelCandidate:
    """
    A model candidate in the ACFA League.

    Tracks deployment stage, metrics, and promotion history.
    """

    model_id: str
    stage: DeploymentStage
    metrics: ModelMetrics
    deployed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted: bool = False
    promotion_decision: Optional[PromotionDecision] = None
    canary_traffic_pct: float = 0.0  # % of traffic (0-100)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "stage": self.stage.value,
            "metrics": asdict(self.metrics),
            "deployed_at": self.deployed_at,
            "promoted": self.promoted,
            "promotion_decision": (
                self.promotion_decision.value if self.promotion_decision else None
            ),
            "canary_traffic_pct": self.canary_traffic_pct,
        }


@dataclass
class LeagueConfig:
    """Configuration for ACFA League orchestrator."""

    # L∞ improvement threshold for promotion (challenger must be X% better)
    linf_improvement_threshold: float = 0.02  # 2% improvement required

    # Minimum L∞ score to be eligible for promotion
    min_linf_score: float = 0.70  # Must achieve 70% L∞

    # Canary deployment traffic percentage
    canary_traffic_start: float = 5.0  # Start with 5% traffic
    canary_traffic_max: float = 50.0  # Max 50% before full promotion

    # Ethics gate: reject if any dimension below this
    ethics_min_threshold: float = 0.60  # 60% minimum on all dimensions

    # Cost threshold: reject if cost exceeds this
    max_cost_ratio: float = 1.5  # Max 150% of champion cost


# ============================================================================
# ACFA League Orchestrator
# ============================================================================


class ACFALeague:
    """
    ACFA League Orchestrator — Champion-Challenger System with L∞ Aggregation.

    Features:
    - Shadow deployment: Challenger runs alongside champion (metrics only)
    - Canary deployment: Challenger serves % of traffic
    - L∞ evaluation: Non-compensatory metric aggregation
    - Ethics gates: ΣEA/LO-14 compliance (fail-closed)
    - Automatic rollback: On ethics violation or performance degradation
    """

    def __init__(
        self,
        config: Optional[LeagueConfig] = None,
        persistence_path: Optional[Path] = None,
    ):
        self.config = config or LeagueConfig()
        self.persistence_path = persistence_path or Path("acfa_league_state.json")

        self.champion: Optional[ModelCandidate] = None
        self.challengers: List[ModelCandidate] = []
        self.history: List[Dict[str, Any]] = []

        # Load state if exists
        self._load_state()

    # ========================================================================
    # Core Methods
    # ========================================================================

    def register_champion(
        self, model_id: str, metrics: ModelMetrics
    ) -> ModelCandidate:
        """
        Register a model as the current champion.

        Args:
            model_id: Unique model identifier
            metrics: Model performance metrics

        Returns:
            ModelCandidate for the champion
        """
        self.champion = ModelCandidate(
            model_id=model_id,
            stage=DeploymentStage.PRODUCTION,
            metrics=metrics,
            promoted=True,
            promotion_decision=PromotionDecision.PROMOTED,
            canary_traffic_pct=100.0,
        )

        self._log_event(
            "champion_registered",
            {
                "model_id": model_id,
                "linf_score": metrics.linf_score(),
                "metrics": asdict(metrics),
            },
        )

        self._save_state()
        return self.champion

    def deploy_challenger(
        self, model_id: str, metrics: ModelMetrics, stage: DeploymentStage = DeploymentStage.SHADOW
    ) -> ModelCandidate:
        """
        Deploy a new challenger model in shadow or canary mode.

        Args:
            model_id: Unique model identifier
            metrics: Model performance metrics
            stage: Deployment stage (default: shadow)

        Returns:
            ModelCandidate for the challenger
        """
        challenger = ModelCandidate(
            model_id=model_id,
            stage=stage,
            metrics=metrics,
            canary_traffic_pct=(
                self.config.canary_traffic_start
                if stage == DeploymentStage.CANARY
                else 0.0
            ),
        )

        self.challengers.append(challenger)

        self._log_event(
            "challenger_deployed",
            {
                "model_id": model_id,
                "stage": stage.value,
                "linf_score": metrics.linf_score(),
                "metrics": asdict(metrics),
            },
        )

        self._save_state()
        return challenger

    def evaluate_challenger(self, model_id: str) -> PromotionDecision:
        """
        Evaluate a challenger for promotion to champion.

        Evaluation criteria:
        1. Ethics gate: All dimensions ≥ ethics_min_threshold
        2. Cost gate: Cost ≤ max_cost_ratio * champion.cost
        3. L∞ improvement: challenger.L∞ ≥ champion.L∞ + improvement_threshold
        4. Minimum L∞: challenger.L∞ ≥ min_linf_score

        Returns:
            PromotionDecision (PROMOTED, REJECTED, or ROLLBACK)
        """
        challenger = self._find_challenger(model_id)
        if not challenger:
            return PromotionDecision.REJECTED

        if not self.champion:
            # No champion yet, promote challenger if passes minimum threshold
            if challenger.metrics.linf_score() >= self.config.min_linf_score:
                return PromotionDecision.PROMOTED
            return PromotionDecision.REJECTED

        # 1. Ethics gate (fail-closed)
        ethics_check = self._check_ethics_gate(challenger.metrics)
        if not ethics_check:
            self._log_event(
                "promotion_rejected",
                {
                    "model_id": model_id,
                    "reason": "ethics_gate_failure",
                    "linf_score": challenger.metrics.linf_score(),
                },
            )
            return PromotionDecision.REJECTED

        # 2. Cost gate
        cost_check = self._check_cost_gate(challenger.metrics, self.champion.metrics)
        if not cost_check:
            self._log_event(
                "promotion_rejected",
                {
                    "model_id": model_id,
                    "reason": "cost_gate_failure",
                    "challenger_cost": challenger.metrics.cost,
                    "champion_cost": self.champion.metrics.cost,
                },
            )
            return PromotionDecision.REJECTED

        # 3. Minimum L∞ eligibility check (documented but previously not enforced)
        challenger_linf = challenger.metrics.linf_score()
        if challenger_linf < self.config.min_linf_score:
            self._log_event(
                "promotion_rejected",
                {
                    "model_id": model_id,
                    "reason": "below_min_linf",
                    "linf_score": challenger_linf,
                    "required": self.config.min_linf_score,
                },
            )
            return PromotionDecision.REJECTED

        # 4. L∞ improvement
        champion_linf = self.champion.metrics.linf_score()
        improvement = challenger_linf - champion_linf

        if improvement >= self.config.linf_improvement_threshold:
            self._log_event(
                "promotion_approved",
                {
                    "model_id": model_id,
                    "challenger_linf": challenger_linf,
                    "champion_linf": champion_linf,
                    "improvement": improvement,
                },
            )
            return PromotionDecision.PROMOTED

        # If not enough improvement yet, progress deployment safely (shadow→canary)
        self.advance_canary(model_id)
        self._log_event(
            "promotion_rejected",
            {
                "model_id": model_id,
                "reason": "insufficient_improvement",
                "challenger_linf": challenger_linf,
                "champion_linf": champion_linf,
                "improvement": improvement,
                "required": self.config.linf_improvement_threshold,
            },
        )
        return PromotionDecision.REJECTED

    def promote_challenger(self, model_id: str) -> bool:
        """
        Promote a challenger to champion.

        Args:
            model_id: Challenger model ID

        Returns:
            True if promoted, False otherwise
        """
        decision = self.evaluate_challenger(model_id)

        if decision != PromotionDecision.PROMOTED:
            return False

        challenger = self._find_challenger(model_id)
        if not challenger:
            return False

        # Demote old champion
        if self.champion:
            self.champion.stage = DeploymentStage.SHADOW
            self.champion.canary_traffic_pct = 0.0

        # Promote challenger
        challenger.stage = DeploymentStage.PRODUCTION
        challenger.promoted = True
        challenger.promotion_decision = PromotionDecision.PROMOTED
        challenger.canary_traffic_pct = 100.0

        # Update champion
        old_champion_id = self.champion.model_id if self.champion else None
        self.champion = challenger
        self.challengers.remove(challenger)

        self._log_event(
            "champion_promoted",
            {
                "new_champion": model_id,
                "old_champion": old_champion_id,
                "linf_score": challenger.metrics.linf_score(),
            },
        )

        self._save_state()
        return True

    def rollback(self, reason: str = "manual_rollback") -> bool:
        """
        Rollback to previous champion (if available).

        Args:
            reason: Reason for rollback

        Returns:
            True if rollback successful, False otherwise
        """
        if not self.champion:
            return False

        # Find previous champion in history
        for event in reversed(self.history):
            if event["event"] == "champion_promoted" and event["data"].get("old_champion"):
                old_champion_id = event["data"]["old_champion"]

                # Find old champion in challengers
                for challenger in self.challengers:
                    if challenger.model_id == old_champion_id:
                        # Restore old champion
                        self.champion.stage = DeploymentStage.SHADOW
                        self.champion.canary_traffic_pct = 0.0

                        challenger.stage = DeploymentStage.PRODUCTION
                        challenger.canary_traffic_pct = 100.0

                        old_champion = self.champion
                        self.champion = challenger
                        self.challengers.remove(challenger)
                        self.challengers.append(old_champion)

                        self._log_event(
                            "rollback_executed",
                            {
                                "restored_champion": old_champion_id,
                                "demoted_model": old_champion.model_id,
                                "reason": reason,
                            },
                        )

                        self._save_state()
                        return True

        return False

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "champion": self.champion.to_dict() if self.champion else None,
            "challengers": [c.to_dict() for c in self.challengers],
            "total_models": 1 + len(self.challengers) if self.champion else len(self.challengers),
        }

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard sorted by L∞ score."""
        all_models = []

        if self.champion:
            all_models.append(self.champion)

        all_models.extend(self.challengers)

        # Sort by L∞ score (descending)
        leaderboard = sorted(
            all_models, key=lambda m: m.metrics.linf_score(), reverse=True
        )

        return [
            {
                "rank": i + 1,
                "model_id": model.model_id,
                "stage": model.stage.value,
                "linf_score": model.metrics.linf_score(),
                "promoted": model.promoted,
            }
            for i, model in enumerate(leaderboard)
        ]

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _find_challenger(self, model_id: str) -> Optional[ModelCandidate]:
        """Find challenger by ID."""
        for challenger in self.challengers:
            if challenger.model_id == model_id:
                return challenger
        return None

    def _check_ethics_gate(self, metrics: ModelMetrics) -> bool:
        """
        Check if metrics pass ethics gate.

        All dimensions must be ≥ ethics_min_threshold.
        """
        values = [
            metrics.accuracy,
            metrics.robustness,
            metrics.calibration,
            metrics.fairness,
            metrics.privacy,
        ]
        return all(v >= self.config.ethics_min_threshold for v in values)

    def _check_cost_gate(
        self, challenger_metrics: ModelMetrics, champion_metrics: ModelMetrics
    ) -> bool:
        """
        Check if challenger cost is acceptable.

        Cost must be ≤ max_cost_ratio * champion.cost
        """
        if champion_metrics.cost == 0:
            return True  # No cost constraint

        cost_ratio = challenger_metrics.cost / champion_metrics.cost
        return cost_ratio <= self.config.max_cost_ratio

    def _log_event(self, event: str, data: Dict[str, Any]):
        """Log an event to history."""
        self.history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": event,
                "data": data,
            }
        )

    def _save_state(self):
        """Save league state to disk."""
        state = {
            "champion": self.champion.to_dict() if self.champion else None,
            "challengers": [c.to_dict() for c in self.challengers],
            "history": self.history[-100:],  # Keep last 100 events
            "config": asdict(self.config),
        }

        self.persistence_path.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        """Load league state from disk."""
        if not self.persistence_path.exists():
            return

        try:
            state = json.loads(self.persistence_path.read_text())

            # Restore champion
            if state.get("champion"):
                c = state["champion"]
                self.champion = ModelCandidate(
                    model_id=c["model_id"],
                    stage=DeploymentStage(c["stage"]),
                    metrics=ModelMetrics(**c["metrics"]),
                    deployed_at=c["deployed_at"],
                    promoted=c["promoted"],
                    promotion_decision=(
                        PromotionDecision(c["promotion_decision"])
                        if c["promotion_decision"]
                        else None
                    ),
                    canary_traffic_pct=c["canary_traffic_pct"],
                )

            # Restore challengers
            for c in state.get("challengers", []):
                self.challengers.append(
                    ModelCandidate(
                        model_id=c["model_id"],
                        stage=DeploymentStage(c["stage"]),
                        metrics=ModelMetrics(**c["metrics"]),
                        deployed_at=c["deployed_at"],
                        promoted=c["promoted"],
                        promotion_decision=(
                            PromotionDecision(c["promotion_decision"])
                            if c["promotion_decision"]
                            else None
                        ),
                        canary_traffic_pct=c["canary_traffic_pct"],
                    )
                )

            # Restore history
            self.history = state.get("history", [])

        except Exception:
            # If load fails, start fresh
            pass

    # ========================================================================
    # Deployment Progression Helpers
    # ========================================================================

    def advance_canary(self, model_id: str) -> bool:
        """
        Progress a challenger through deployment stages safely.

        - SHADOW → CANARY (start at canary_traffic_start)
        - CANARY → increase traffic up to canary_traffic_max
        Returns True if any progression occurred.
        """
        challenger = self._find_challenger(model_id)
        if not challenger:
            return False

        progressed = False
        if challenger.stage == DeploymentStage.SHADOW:
            challenger.stage = DeploymentStage.CANARY
            challenger.canary_traffic_pct = self.config.canary_traffic_start
            progressed = True
        elif challenger.stage == DeploymentStage.CANARY:
            new_pct = min(
                self.config.canary_traffic_max,
                max(challenger.canary_traffic_pct * 2.0, self.config.canary_traffic_start),
            )
            if new_pct > challenger.canary_traffic_pct:
                challenger.canary_traffic_pct = new_pct
                progressed = True

        if progressed:
            self._log_event(
                "canary_progressed",
                {
                    "model_id": model_id,
                    "stage": challenger.stage.value,
                    "canary_traffic_pct": challenger.canary_traffic_pct,
                },
            )
            self._save_state()

        return progressed

    def update_challenger_metrics(self, model_id: str, metrics: ModelMetrics) -> bool:
        """Update stored metrics for a challenger and persist state."""
        challenger = self._find_challenger(model_id)
        if not challenger:
            return False
        challenger.metrics = metrics
        self._log_event(
            "challenger_metrics_updated",
            {"model_id": model_id, "linf_score": metrics.linf_score(), "metrics": asdict(metrics)},
        )
        self._save_state()
        return True
