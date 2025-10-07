import random
from dataclasses import dataclass, field
from typing import Any, List, Dict
from copy import deepcopy

Genome = Any
Metrics = Dict[str, Any]

@dataclass
class MetaGenome:
    mut_rate: float = 0.10
    cx_rate: float = 0.80
    mut_scale: float = 0.25

@dataclass
class Individual:
    genome: Genome
    meta_genome: MetaGenome = field(default_factory=MetaGenome)
    metrics: Metrics = field(default_factory=dict)
    behavior: List[float] = field(default_factory=list)
    score: float = 0.0

    def clone(self) -> "Individual":
        return Individual(genome=deepcopy(self.genome),
                          meta_genome=deepcopy(self.meta_genome),
                          metrics=self.metrics.copy(),
                          behavior=list(self.behavior),
                          score=self.score)

class Population:
    def __init__(self, size: int, plugin: "BasePlugin", seed: int = 42):
        self.rng = random.Random(seed)
        self.plugin = plugin
        self.members: List[Individual] = [plugin.create_individual(self.rng) for _ in range(size)]

    def tournament_select(self, k: int = 3) -> Individual:
        pool = self.rng.sample(self.members, k=min(k, len(self.members)))
        return max(pool, key=lambda ind: ind.score)

    def _evolve_meta_genome(self, p1_meta: MetaGenome, p2_meta: MetaGenome, meta_learning_rate: float) -> MetaGenome:
        child_meta = MetaGenome(
            mut_rate=(p1_meta.mut_rate + p2_meta.mut_rate) / 2,
            cx_rate=(p1_meta.cx_rate + p2_meta.cx_rate) / 2,
            mut_scale=(p1_meta.mut_scale + p2_meta.mut_scale) / 2,
        )
        mutation_magnitude = 0.02 * meta_learning_rate
        child_meta.mut_rate = max(0.01, min(0.9, child_meta.mut_rate + self.rng.gauss(0, mutation_magnitude)))
        child_meta.cx_rate = max(0.1, min(0.95, child_meta.cx_rate + self.rng.gauss(0, mutation_magnitude)))
        child_meta.mut_scale = max(0.01, min(1.0, child_meta.mut_scale + self.rng.gauss(0, mutation_magnitude / 2)))
        return child_meta

    def make_offspring(self, meta_learning_rate: float):
        new_pop = []
        # In a real scenario, you might have a more sophisticated elitism strategy.
        # For simplicity, we'll just carry over the best individual.
        if self.members:
            elite = max(self.members, key=lambda ind: ind.score)
            new_pop.append(elite.clone())

        while len(new_pop) < len(self.members):
            p1 = self.tournament_select()
            p2 = self.tournament_select()
            
            # Placeholder for plugin reference
            # In the full engine, the plugin would be passed in.
            # For now, we assume it exists.
            plugin = getattr(self, 'plugin', None)
            if not plugin:
                # This is a fallback and shouldn't happen in the engine context.
                # Create a dummy offspring if no plugin is found.
                new_pop.append(p1.clone())
                continue

            child_meta_genome = self._evolve_meta_genome(p1.meta_genome, p2.meta_genome, meta_learning_rate)
            
            if self.rng.random() < child_meta_genome.cx_rate:
                child_genome = plugin.crossover(p1.genome, p2.genome, self.rng)
            else:
                child_genome = deepcopy(p1.genome)
            
            child_genome = plugin.mutate(
                child_genome,
                rate=child_meta_genome.mut_rate,
                scale=child_meta_genome.mut_scale,
                rng=self.rng
            )
            new_pop.append(Individual(genome=child_genome, meta_genome=child_meta_genome))
        self.members = new_pop
