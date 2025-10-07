import random, time, hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable

Genome = Dict[str, float]; Metrics = Dict[str, Any]
def _hash(x:str)->str: return hashlib.sha256(x.encode('utf-8')).hexdigest()

@dataclass
class Individual:
    genome: Genome
    metrics: Metrics = field(default_factory=dict)
    behavior: List[float] = field(default_factory=list)
    score: float = 0.0
    # genealogia
    iid: str = field(default_factory=lambda: _hash(f"{time.time_ns()}"))
    parents: List[str] = field(default_factory=list)
    born_cycle: int = 0
    born_gen: int = 0

    def clone(self)->"Individual":
        c = Individual(self.genome.copy(), self.metrics.copy(), list(self.behavior), self.score)
        c.parents = list(self.parents); c.born_cycle=self.born_cycle; c.born_gen=self.born_gen
        return c

def uniform_crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    keys=set(a.keys())|set(b.keys()); child={}
    for k in keys:
        child[k] = (a.get(k,0.0) if rng.random()<0.5 else b.get(k,0.0))
    return child

def gaussian_mutation(g: Genome, rate: float, scale: float, rng: random.Random) -> Genome:
    out=g.copy()
    for k,v in list(out.items()):
        if rng.random()<rate: out[k]=v + rng.gauss(0.0, scale)
    # chance pequena de "novo gene"
    if rng.random()< rate*0.15:
        k=f"g{rng.randint(1,1_000_000)}"; out[k]=rng.gauss(0.0, scale*2)
    return out

class Population:
    def __init__(self, size:int, init_genome_fn:Callable[[random.Random], Genome], seed:int=42):
        assert size>=2
        self.rng = random.Random(seed)
        self.members: List[Individual] = []
        for _ in range(size):
            g = init_genome_fn(self.rng); self.members.append(Individual(g))
    def tournament_select(self,k:int=3)->Individual:
        pool=self.rng.sample(self.members, k=min(k,len(self.members)))
        return max(pool, key=lambda ind: ind.score)
    def make_offspring(self, cx_rate:float, mut_rate:float, mut_scale:float,
                       cycle:int, gen:int,
                       init_genome_fn:Callable[[random.Random], Genome])->List[Individual]:
        new=[]; elite=max(self.members, key=lambda i:i.score); new.append(elite.clone())
        while len(new)<len(self.members):
            if self.rng.random()<cx_rate:
                p1=self.tournament_select(); p2=self.tournament_select()
                genome=uniform_crossover(p1.genome, p2.genome, self.rng)
                parents=[p1.iid, p2.iid]
            else:
                p=self.tournament_select(); genome=p.genome.copy(); parents=[p.iid]
            genome=gaussian_mutation(genome, rate=mut_rate, scale=mut_scale, rng=self.rng)
            child=Individual(genome); child.parents=parents; child.born_cycle=cycle; child.born_gen=gen
            new.append(child)
        self.members=new; return self.members
