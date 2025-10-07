from typing import Dict, Any
from .omega_equations import linf_score, compute_caos_plus, sr_omega_score

def harmonic_mean(values, weights=None, eps=1e-9):
    if not values: return 0.0
    if weights is None: weights = [1.0] * len(values)
    num = sum(weights)
    denom = sum(w / max(eps, float(v)) for v, w in zip(values, weights))
    return num / max(eps, denom)

def calculate_phi_score(metrics: Dict[str, Any]) -> float:
    raw_metrics = metrics.get("raw_metrics", {})
    metric_weights = metrics.get("metric_weights", {})
    cost = metrics.get("cost", 0.0)
    linf = linf_score(raw_metrics, metric_weights, cost)
    
    novelty = metrics.get("novelty", 0.0)
    sr = sr_omega_score(metrics.get("reflection_quality", 0.0), metrics.get("uncertainty", 1.0))
    
    ood_generalization = metrics.get("ood_generalization", 1.0)
    equity_score = metrics.get("equity_score", 1.0)
    
    value_portfolio = [linf, ood_generalization, equity_score]
    value_weights = [3.0, 1.5, 1.0]
    base_value = harmonic_mean(value_portfolio, value_weights)
    
    expansion_bonus = (novelty * 0.15) + (sr * 0.1)
    
    caos = compute_caos_plus(
        metrics.get("coherence", 0.8), metrics.get("adaptability", 0.7),
        metrics.get("opportunity", 0.9), metrics.get("security", 1.0)
    )
    cost_penalty = float(metrics.get("cost_penalty", 1.0))
    ethics_pass = 1.0 if metrics.get("ethics_pass", True) else 0.0

    final_score = (base_value + expansion_bonus) * caos * cost_penalty * ethics_pass
    return max(0.0, final_score)
