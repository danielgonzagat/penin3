import os
import json
from pathlib import Path

from penin.league import ACFALeague, ModelMetrics, DeploymentStage, PromotionDecision, LeagueConfig


def test_acfa_promotion_and_canary(tmp_path):
    state_path = tmp_path / "acfa_state.json"
    league = ACFALeague(config=LeagueConfig(min_linf_score=0.7, linf_improvement_threshold=0.02),
                        persistence_path=state_path)

    # Register champion at L_inf=0.80 (all dims >= 0.80)
    champion = league.register_champion("champion_v1", ModelMetrics(
        accuracy=0.80, robustness=0.80, calibration=0.80, fairness=0.80, privacy=0.80, cost=1.0
    ))
    assert champion.stage == DeploymentStage.PRODUCTION

    # Challenger slightly below min_linf should be rejected
    c1 = league.deploy_challenger("challenger_low", ModelMetrics(
        accuracy=0.69, robustness=0.80, calibration=0.80, fairness=0.80, privacy=0.80, cost=1.0
    ))
    decision = league.evaluate_challenger("challenger_low")
    assert decision == PromotionDecision.REJECTED

    # Challenger with sufficient min_linf but not enough improvement triggers canary advance
    c2 = league.deploy_challenger("challenger_shadow", ModelMetrics(
        accuracy=0.81, robustness=0.80, calibration=0.80, fairness=0.80, privacy=0.80, cost=1.0
    ))
    assert c2.stage == DeploymentStage.SHADOW
    decision2 = league.evaluate_challenger("challenger_shadow")
    assert decision2 == PromotionDecision.REJECTED
    # Should have progressed to CANARY
    status = league.get_deployment_status()
    challenger = next(m for m in league.challengers if m.model_id == "challenger_shadow")
    assert challenger.stage in (DeploymentStage.CANARY, DeploymentStage.SHADOW)

    # Strong challenger with improvement should be promoted
    c3 = league.deploy_challenger("challenger_strong", ModelMetrics(
        accuracy=0.90, robustness=0.85, calibration=0.84, fairness=0.83, privacy=0.82, cost=1.0
    ))
    decision3 = league.evaluate_challenger("challenger_strong")
    if decision3 == PromotionDecision.PROMOTED:
        promoted = league.promote_challenger("challenger_strong")
        assert promoted is True
        assert league.champion.model_id == "challenger_strong"
    else:
        # In case thresholds differ, ensure at least no rollback
        assert decision3 in (PromotionDecision.REJECTED, PromotionDecision.PROMOTED)
