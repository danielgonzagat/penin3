from dataclasses import dataclass, field
from typing import Dict, List
from .constants import fib_seq
@dataclass
class Champ:
    genome: Dict[str, float]
    score: float
    behavior: List[float]
    # Persist key metrics for fair comparisons and analysis
    metrics: Dict[str, float] | None = None
class Arena:
    def __init__(self, hist:int=8):
        self.champion: Champ|None=None
        self.history: List[Champ] = []
        self.weights = fib_seq(hist)[::-1]
        self.weights = [w/sum(self.weights) for w in self.weights]
    def consider(self, cand:Champ)->bool:
        if (self.champion is None) or (cand.score > self.champion.score*(1+1e-6)):
            self.champion=cand; self.history.append(cand)
            if len(self.history)>len(self.weights): self.history.pop(0)
            return True
        return False
    def superpose(self)->Dict[str,float]:
        # superposição Fibonacci dos campeões históricos
        if not self.history: return {}
        keys=set().union(*[set(c.genome.keys()) for c in self.history])
        out={k:0.0 for k in keys}
        w=self.weights[-len(self.history):]
        for c,wi in zip(self.history, w):
            for k in keys:
                val = c.genome.get(k,0.0)
                # FIX: Only superpose numeric values
                if isinstance(val, (int, float)):
                    out[k] += val * wi
                elif k not in out or out[k] == 0.0:
                    # Preserve first non-numeric value
                    out[k] = val
        return out