from dataclasses import dataclass
import math
from .constants import fib, fib_list

@dataclass
class FBudget:
    generations: int
    checkpoint: bool
    mut_rate: float
    cx_rate: float
    cycle_resource_budget: float
    promotion_pressure: float
    meta_learning_rate: float

class FClock:
    def __init__(self, max_cycles: int, base_mut: float = 0.08, base_cx: float = 0.75, base_budget: float = 1000.0):
        self.max_cycles = max_cycles
        self.seq = fib_list(max_cycles + 5)
        self.base_mut = base_mut
        self.base_cx = base_cx
        self.base_budget = base_budget
        self.max_f_in_run = fib(max_cycles) if max_cycles > 0 else 1

    def budget_for_cycle(self, cycle: int) -> FBudget:
        n = max(1, cycle)
        f = self.seq[n - 1] if n - 1 < len(self.seq) else fib(n)
        rhythmic_progress = math.log1p(f) / math.log1p(self.max_f_in_run) if self.max_f_in_run > 1 else 1
        promotion_pressure = 1.15 - (0.12 * rhythmic_progress)
        meta_learning_rate = 0.2 + (0.8 * rhythmic_progress)
        budget = self.base_budget * (1 + math.log1p(f))
        
        return FBudget(
            generations=min(64, max(5, f)),
            checkpoint=(f % 5 == 0) or (n in {1,2}),
            mut_rate=min(0.35, max(0.02, self.base_mut * (1.0 + (f % 8) / 12.0))),
            cx_rate=min(0.95, max(0.20, self.base_cx * (1.0 - (1.0 / (1.0 + f))))),
            cycle_resource_budget=budget,
            promotion_pressure=promotion_pressure,
            meta_learning_rate=meta_learning_rate
        )
