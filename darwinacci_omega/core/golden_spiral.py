import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class SpiralBin:
    best_score: float = float("-inf")
    behavior: List[float] = field(default_factory=list)
    genome: Dict[str, float] = field(default_factory=dict)  # snapshot opcional

class GoldenSpiralArchive:
    """
    Nichos angulares pela razão áurea (aprox. CVT-lite sem numpy).
    Usa ângulo de (x,y) e 89 bins (número de Fibonacci) por padrão.
    """
    def __init__(self, bins:int=89):
        self.bins=bins
        self.archive: Dict[int, SpiralBin] = {}
        # Light-weight skills cache: top K genomes/behaviors for biasing/recall
        self._skills_cache: List[Tuple[int, SpiralBin]] = []
    @staticmethod
    def _angle(b:List[float])->float:
        if not b: return 0.0
        x=b[0]; y=b[1] if len(b)>1 else 0.0
        return math.atan2(y,x) % (2*math.pi)
    def _bin(self, behavior:List[float])->int:
        theta=self._angle(behavior)
        return int((theta/(2*math.pi))*self.bins) % self.bins
    def add(self, behavior:List[float], score:float, genome:Dict[str, float]|None=None):
        idx=self._bin(behavior)
        cell=self.archive.get(idx, SpiralBin())
        if score>cell.best_score:
            cell.best_score=score
            cell.behavior=list(behavior)
            if genome is not None:
                try:
                    cell.genome=dict(genome)
                except Exception:
                    cell.genome={}
        self.archive[idx]=cell
        # Update skills cache lazily (keep small)
        try:
            self._skills_cache = self.bests()[: min(32, len(self.archive))]
        except Exception:
            pass
    def coverage(self)->float:
        return len(self.archive)/float(self.bins)
    def bests(self)->List[Tuple[int,SpiralBin]]:
        return sorted(self.archive.items(), key=lambda kv: kv[1].best_score, reverse=True)

    # -------- Skills helpers --------
    def best_skills(self, k:int=8)->List[Dict[str, float]]:
        """Return up to k top genome snapshots suitable as 'skills' seeds."""
        out: List[Dict[str, float]] = []
        src = self._skills_cache if self._skills_cache else self.bests()
        for _, cell in src[:k]:
            if cell.genome:
                out.append(dict(cell.genome))
            else:
                # reconstruct minimal skill from behavior
                b0 = float(cell.behavior[0]) if len(cell.behavior)>0 else 0.25
                b1 = float(cell.behavior[1]) if len(cell.behavior)>1 else 1.0
                out.append({'skill_b0': b0, 'skill_b1': b1})
        return out

    def export_skills_json(self, path: str, k:int=16) -> bool:
        """Export top skills to JSON for external tooling. Returns success flag."""
        try:
            import json, os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            skills = self.best_skills(k=k)
            with open(path, 'w') as f:
                json.dump({'skills': skills}, f)
            return True
        except Exception:
            return False