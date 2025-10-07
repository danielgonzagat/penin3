import random
from .population import Individual
def godel_kick(pop, rng:random.Random, severity:float=0.25, top_k:int=3):
    if not pop.members: return
    sorted_pop = sorted(pop.members, key=lambda i:i.score, reverse=True)
    for i in range(min(top_k, len(sorted_pop))):
        ind = sorted_pop[i]
        # perturba genes chaves
        for k in list(ind.genome.keys()):
            if rng.random()<severity: ind.genome[k] += rng.gauss(0.0, severity)
        # injeta gene novo ocasionalmente
        if rng.random()<0.33:
            ind.genome[f"axiom_{rng.randint(1,9)}"] = rng.gauss(0.0, severity*2)
