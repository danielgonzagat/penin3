from typing import Dict, Any
import math

def linf_score(metrics: Dict[str, float], weights: Dict[str, float], cost: float = 0.0) -> float:
    if not metrics or not weights:
        return 0.0
    min_weighted_metric = float('inf')
    for key, weight in weights.items():
        metric_val = metrics.get(key, 0.0)
        if weight > 0:
            min_weighted_metric = min(min_weighted_metric, metric_val / weight)
    return max(0.0, min_weighted_metric - cost)

def compute_caos_plus(c: float, a: float, o: float, s: float) -> float:
    c, a, o, s = [max(0.0, min(1.0, val)) for val in [c, a, o, s]]
    stability = (c * s)**0.5
    exploration = (a * o)**0.5
    return (stability * exploration)

def sr_omega_score(reflection_quality: float, uncertainty: float) -> float:
    effective_uncertainty = max(0, uncertainty - reflection_quality * 0.5)
    return max(0, 1.0 - effective_uncertainty)
