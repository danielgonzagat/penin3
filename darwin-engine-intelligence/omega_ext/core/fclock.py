from dataclasses import dataclass
from .constants import fib, fib_list
@dataclass
class FBudget: generations:int; checkpoint:bool; mut_rate:float; cx_rate:float
class FClock:
    def __init__(self, max_cycles:int, base_mut:float=0.08, base_cx:float=0.75):
        self.seq = fib_list(max_cycles + 8); self.base_mut=base_mut; self.base_cx=base_cx
    def budget_for_cycle(self, cycle:int)->FBudget:
        n=max(1,cycle); f = self.seq[n-1] if n-1 < len(self.seq) else fib(n)
        gens = min(96, max(6, f))
        mut  = min(0.40, max(0.02, self.base_mut*(1.0+ (f%8)/12.0)))
        cx   = min(0.95, max(0.20, self.base_cx *(1.0- (1.0/(1.0+f)))))
        checkpoint = (f%5==0) or (n in {1,2})
        return FBudget(generations=gens, checkpoint=checkpoint, mut_rate=mut, cx_rate=cx)
