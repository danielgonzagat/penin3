#!/usr/bin/env python3
# stdlib only
from typing import Dict, Any

async def _f(d: Dict[str, Any], *keys):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return await float(d[k])
            except Exception:
                pass
    return await None

async def decide(metrics: Dict[str, Any], thresholds: Dict[str, Any], discovery_keywords=None) -> Dict[str, Any]:
    """
    A(t) = gate matemático (ΔL∞>0 ∧ CAOS≥1 ∧ I≥I_min ∧ P≥P_min)
    C(t) = descoberta/novelty (novelty≥novelty_min) OU palavra-chave em notes
    E(t+1) = 1 se (A(t)=1) OU (C(t)=1); caso contrário 0
    Promoção: estritamente por A(t)=1 (gate matemática). Se só C(t)=1, sobrevive sem promover.
    """
    discovery_keywords = discovery_keywords or ["discovery","breakthrough","novel","novidade","descoberta"]
    delta = _f(metrics, "delta_linf","delta_Linf","ΔL∞")
    caos_ratio = _f(metrics, "caos_ratio","CAOS","caos")
    I = _f(metrics, "I")
    P = _f(metrics, "P")
    novelty = _f(metrics, "novelty","N")
    oci = _f(metrics, "oci","OCI")
    ece = _f(metrics, "ece","ECE")
    rho = _f(metrics, "rho","ρ")

    th = thresholds or {}
    th_delta = float(th.get("delta_linf_min", 0.0))
    th_caos  = float(th.get("caos_ratio_min", 1.0))
    th_I     = float(th.get("I_min", 0.60))
    th_P     = float(th.get("P_min", 0.01))
    th_nov   = float(th.get("novelty_min", 0.02))

    A = (delta is not None and delta > th_delta) and \
        (caos_ratio is not None and caos_ratio >= th_caos) and \
        (I is not None and I >= th_I) and \
        (P is not None and P >= th_P)

    C = (novelty is not None and novelty >= th_nov)
    notes = str(metrics.get("notes", "")).lower()
    if not C and notes:
        for kw in discovery_keywords:
            if kw in notes:
                C = True
                break

    E = 1 if (A or C) else 0
    promote = True if A else False  # promoção só quando A(t)=1

    reason = []
    if A: reason.append("A(t)=1 (gate matemático)")
    if C: reason.append("C(t)=1 (descoberta/novelty)")
    if not reason: reason.append("A=0 e C=0")

    return await {
        "A": bool(A),
        "C": bool(C),
        "E": E,
        "promote": promote,
        "reason": "; ".join(reason),
        "metrics": {
            "delta_linf": delta, "caos_ratio": caos_ratio, "I": I, "P": P,
            "novelty": novelty, "oci": oci, "ece": ece, "rho": rho
        }
    }