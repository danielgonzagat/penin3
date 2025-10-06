import random
from typing import Dict, List
Genome = Dict[str, float]

def tournament(pop, k:int, key, rng: random.Random):
    """Deterministic tournament using provided RNG."""
    k = min(k, len(pop))
    pool = rng.sample(pop, k) if k > 0 else []
    return max(pool, key=key) if pool else max(pop, key=key)

def uniform_cx(a:Genome,b:Genome,rng)->Genome:
    keys=set(a.keys())|set(b.keys()); c={}
    for k in keys: c[k]=a.get(k,0.0) if rng.random()<0.5 else b.get(k,0.0)
    return c

def gaussian_mut(g:Genome, rate:float, scale:float, rng)->Genome:
    out=g.copy()
    for k,v in list(out.items()):
        # FIX: Only mutate numeric values (skip strings)
        if isinstance(v, (int, float)) and rng.random()<rate:
            out[k]=v + rng.gauss(0.0, scale)
    if rng.random()<rate*0.12: out[f"g{rng.randint(1,1_000_000)}"]=rng.gauss(0.0, scale*2)
    return out

def prune_genes(g: Genome, max_genes: int, rng: random.Random) -> Genome:
    """
    Limit genome size by pruning least useful or auxiliary keys (FIX BUG #4).
    Preference: remove keys starting with 'axiom_' or 'g<id>' first, then smallest magnitude.
    GARANTIA: sempre retorna genoma <= max_genes
    """
    if max_genes <= 0 or len(g) <= max_genes:
        return g
    
    keys = list(g.keys())
    # Priority 1: auxiliary keys
    aux = [k for k in keys if k.startswith('axiom_') or (k.startswith('g') and k[1:].isdigit())]
    core = [k for k in keys if k not in aux]
    
    pruned = dict(g)
    excess = len(g) - max_genes
    
    # Remove from aux first
    if excess > 0 and aux:
        drop_count = min(len(aux), excess)
        drop = rng.sample(aux, drop_count)
        for k in drop:
            pruned.pop(k, None)
        excess = max(0, len(pruned) - max_genes)
    
    # If still exceeds, remove smallest magnitude among core
    if excess > 0 and core:
        ranked = sorted(core, key=lambda k: abs(float(pruned.get(k, 0.0))))
        drop_count = min(len(core), excess)
        drop = ranked[:drop_count]
        for k in drop:
            pruned.pop(k, None)
    
    # FINAL GUARANTEE: force limit if still exceeds (last resort)
    if len(pruned) > max_genes:
        # Keep highest magnitude keys
        keys_sorted = sorted(pruned.keys(), key=lambda k: abs(float(pruned.get(k, 0.0))), reverse=True)
        pruned = {k: pruned[k] for k in keys_sorted[:max_genes]}
    
    return pruned