import random
from typing import Callable, Dict, Any, List, Optional
from .fclock import FClock
from .population import Population, Individual
from .novelty import NoveltyArchive
from .fitness import aggregate_multiobjective
from .gates import SigmaGuard
from .worm import WORMLedger
from .champion import ChampionArena
from .meta_evolution import MetaEvolution
from .godel import godel_kick

EvalFn = Callable[[Dict[str,float], random.Random], Dict[str,Any]]
InitFn = Callable[[random.Random], Dict[str,float]]
BreedFn = Callable[[Population, float, float, float, int, int, InitFn], List[Individual]]
CanaryFn = Callable[[Individual], bool]

class DarwinOmegaBridge:
    def __init__(self, init_genome_fn:InitFn, eval_fn:EvalFn, *,
                 seed:int=123, max_cycles:int=7,
                 thresholds:Optional[Dict[str,float]]=None,
                 canary_fn:Optional[CanaryFn]=None,
                 breed_fn:Optional[BreedFn]=None):
        self.rng=random.Random(seed); self.eval_fn=eval_fn; self.init_fn=init_genome_fn
        self.clock=FClock(max_cycles=max_cycles); self.meta=MetaEvolution()
        self.guard=SigmaGuard(thresholds or {"ece_max":0.1,"rho_bias_max":1.05,"rho_max":0.99})
        self.ledger=WORMLedger(); self.arena=ChampionArena(canary_fn=canary_fn)
        self.archive=NoveltyArchive(k=7,max_size=2000); self.population:Optional[Population]=None
        self.cycle=0; self.breed_fn=breed_fn

    def _ensure_pop(self, size:int):
        if self.population is None or len(self.population.members)!=size:
            self.population = Population(size, self.init_fn, seed=self.rng.randint(1,1_000_000))

    def _evaluate(self)->float:
        nov_sum=0.0
        for ind in self.population.members:
            m=self.eval_fn(ind.genome, self.rng)  # usa teu Darwin para medir
            ind.metrics=m
            behavior=m.get("behavior") or list(ind.genome.values())
            ind.behavior=behavior
            nov=self.archive.score(behavior); ind.metrics["novelty"]=nov; nov_sum+=nov
            ind.score=aggregate_multiobjective(ind.metrics)
        mean_novelty=nov_sum/max(1,len(self.population.members))
        for ind in self.population.members: self.archive.add(ind.behavior)
        return mean_novelty

    def run(self, max_cycles:int=7):
        last_best=None; last_best_score=0.0
        for c in range(1, max_cycles+1):
            self.cycle=c; budget=self.clock.budget_for_cycle(c)
            progress_delta=max(0.0,(last_best.score-last_best_score)) if last_best else 0.0
            params=self.meta.step(progress_delta=progress_delta, novelty_mean=0.0, f_step=budget.generations)
            self._ensure_pop(params.pop_size)

            mean_novelty=0.0
            for g in range(budget.generations):
                mean_novelty=self._evaluate()
                # anti-estagnação: se não melhora  e novelty baixo → Gödel kick
                if g>2 and mean_novelty<0.05: godel_kick(self.population, self.rng, severity=0.25, top_k=3)
                # reprodução
                if self.breed_fn:
                    self.population.members = self.breed_fn(self.population, params.cx_rate, params.mut_rate,
                                                            params.mut_scale, c, g, self.init_fn)
                else:
                    self.population.make_offspring(params.cx_rate, params.mut_rate, params.mut_scale, c, g, self.init_fn)

            mean_novelty=self._evaluate(); best=max(self.population.members, key=lambda i:i.score)
            ok, reasons=self.guard.evaluate(best.metrics); best.metrics["ethics_pass"]=ok
            accepted=False
            if ok: accepted=self.arena.consider(best)
            entry={"cycle":c,"accepted":bool(accepted),"checkpoint":bool(budget.checkpoint),
                   "best_score":float(best.score),"mean_novelty":float(mean_novelty),
                   "pop_size":int(params.pop_size),"mut_rate":float(params.mut_rate),
                   "cx_rate":float(params.cx_rate),"mut_scale":float(params.mut_scale),
                   "reasons":{k:float(v) for k,v in reasons.items()},
                   "best_iid":best.iid,"parents":best.parents,"metrics":{k:(float(v) if isinstance(v,(int,float)) else v) for k,v in best.metrics.items()}}
            h=self.ledger.append(entry)
            print(f"[Ω-bridge] ciclo={c:02d} best={best.score:.4f} accept={accepted} pop={params.pop_size} "
                  f"mut={params.mut_rate:.3f} cx={params.cx_rate:.3f} nov={mean_novelty:.3f} hash={h[:10]}")
            last_best_score = last_best.score if last_best else 0.0; last_best=best
        print("\n[Ω-bridge] campeão:", (self.arena.champion.score if self.arena.champion else 0.0))
        return self.arena.champion
