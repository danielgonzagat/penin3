from typing import Optional, Callable
from .population import Individual
class ChampionArena:
    def __init__(self, epsilon:float=1e-6, canary_fn:Optional[Callable[[Individual],bool]]=None):
        self.champion: Optional[Individual]=None; self.epsilon=epsilon; self.canary_fn=canary_fn
    def consider(self, challenger:Individual)->bool:
        passes = (self.champion is None) or (challenger.score > (self.champion.score + self.epsilon))
        if not passes: return False
        if self.canary_fn and (not self.canary_fn(challenger)): return False
        self.champion = challenger.clone()
        return True
