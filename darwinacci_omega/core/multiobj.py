from typing import List, Dict, Any
def hmean(vals, w=None, eps=1e-9):
    if not vals: return 0.0
    if w is None: w=[1.0]*len(vals)
    num=sum(w); den=0.0
    for v,wi in zip(vals,w): v=max(eps,float(v)); den+= wi/v
    return num/max(eps,den)

def agg(m:Dict[str,Any])->float:
    obj=float(m.get("objective",0.0))
    linf=float(m.get("linf",0.0))
    nov =float(m.get("novelty",0.0))
    rob =float(m.get("robustness",1.0))
    caos=float(m.get("caos_plus",1.0))
    cost=float(m.get("cost_penalty",1.0))
    base=hmean([max(0,obj),max(0,linf),max(0,nov*0.5),max(0,rob)],[2,2,1,1])
    return base * max(0.0,min(1.0,cost)) * max(0.2,min(2.0,caos))

def dominates(a:Dict[str,float], b:Dict[str,float], keys=("objective","novelty"))->bool:
    better=False
    for k in keys:
        if a.get(k,0.0) < b.get(k,0.0): return False
        if a.get(k,0.0) > b.get(k,0.0): better=True
    return better