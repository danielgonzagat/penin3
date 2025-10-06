from dataclasses import dataclass
from .constants import fib_seq, PHI, clamp
@dataclass
class FBudget: 
    generations:int; checkpoint:bool; mut:float; cx:float; elite:int
class TimeCrystal:
    def __init__(self, max_cycles:int, base_mut=0.08, base_cx=0.75, base_elite=4):
        self.max_cycles = max_cycles  # FIX: Store max_cycles as attribute
        self.seq=fib_seq(max_cycles+8)
        self.base_mut=base_mut; self.base_cx=base_cx; self.base_elite=base_elite
        self.phase=0.0; self.inc=1/PHI  # quase-periódico
    def budget(self, cycle:int)->FBudget:
        f=self.seq[cycle-1] if cycle-1<len(self.seq) else self.seq[-1]
        self.phase=(self.phase+self.inc)%1.0
        gens=max(6, min(96, f))
        mut=clamp(self.base_mut*(1+0.5*self.phase), 0.02, 0.45)
        cx =clamp(self.base_cx *(1-0.3*self.phase), 0.20, 0.95)
        elite=max(2, int(self.base_elite*(1+0.2*self.phase)))
        checkpoint=(f%5==0) or (cycle in (1,2))
        return FBudget(generations=gens, checkpoint=checkpoint, mut=mut, cx=cx, elite=elite)