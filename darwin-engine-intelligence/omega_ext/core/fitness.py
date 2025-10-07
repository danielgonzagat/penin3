from typing import Dict, Any
def harmonic_mean(values, weights=None, eps=1e-9):
    if not values: return 0.0
    if weights is None: weights=[1.0]*len(values)
    num=sum(weights); denom=0.0
    for v,w in zip(values,weights):
        v=max(eps,float(v)); denom += w / v
    return num/max(eps,denom)
def aggregate_multiobjective(m:Dict[str,Any])->float:
    objective=float(m.get("objective",0.0))
    linf=float(m.get("linf",0.0))
    caos=float(m.get("caos_plus",1.0))
    novelty=float(m.get("novelty",0.0))
    robust=float(m.get("robustness",1.0))
    cost=float(m.get("cost_penalty",1.0))
    ethics=1.0 if m.get("ethics_pass",True) else 0.0
    base=harmonic_mean([max(0.0,objective), max(0.0,linf), max(0.0,novelty*0.5), max(0.0,robust)],
                       weights=[2.0,2.0,1.0,1.0])
    return base * max(0.0,min(1.0,cost)) * ethics * max(0.2,min(2.0,caos))
