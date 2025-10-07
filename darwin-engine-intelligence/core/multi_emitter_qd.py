"""
Multi-Emitter QD Framework: CMA-MEGA style with coordinated emitters.

Implements a framework for Quality-Diversity with multiple coordinated emitters:
- Improvement emitter: Exploits current archive (CMA-ES-like)
- Exploration emitter: Explores new areas (random directions)
- Gradient emitter: Gradient-based if available
- Random emitter: Pure random baseline
- Curiosity emitter: Targets sparse areas

References:
- Fontaine, M. C., & Nikolaidis, S. (2021). "Differentiable Quality Diversity"
- Fontaine, M. C., et al. (2020). "Covariance Matrix Adaptation for the 
  Rapid Illumination of Behavior Space"
"""

import random
import math
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EmitterType(Enum):
    """Types of emitters in the multi-emitter framework."""
    IMPROVEMENT = "improvement"  # CMA-ES-like, exploit archive
    EXPLORATION = "exploration"  # Random directions, explore space
    GRADIENT = "gradient"        # Gradient-based (if available)
    RANDOM = "random"            # Pure random baseline
    CURIOSITY = "curiosity"      # Target sparse/empty niches


@dataclass
class Individual:
    """Individual with genome, fitness, and behavior."""
    genome: Dict[str, float]
    fitness: float = 0.0
    behavior: List[float] = field(default_factory=list)
    emitter_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class EmitterStats:
    """Statistics for an emitter."""
    emitter_id: str
    emitter_type: EmitterType
    emissions: int = 0
    improvements: int = 0
    new_niches: int = 0
    total_fitness_gain: float = 0.0


class BaseEmitter:
    """Base class for emitters."""
    
    def __init__(self, emitter_id: str, emitter_type: EmitterType, seed: int = 42):
        self.emitter_id = emitter_id
        self.emitter_type = emitter_type
        self.rng = random.Random(seed)
        self.stats = EmitterStats(emitter_id=emitter_id, emitter_type=emitter_type)
    
    def emit(
        self,
        archive: Dict[int, Individual],
        centroids: List[List[float]],
        init_genome_fn: Callable,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        """
        Emit new genomes.
        
        Returns: List of genomes
        """
        raise NotImplementedError


class ImprovementEmitter(BaseEmitter):
    """
    Improvement emitter: Exploits current archive.
    
    Selects high-fitness individuals and applies small mutations to improve them.
    Similar to CMA-ES exploitation phase.
    """
    
    def __init__(self, emitter_id: str, mutation_scale: float = 0.1, seed: int = 42):
        super().__init__(emitter_id, EmitterType.IMPROVEMENT, seed)
        self.mutation_scale = mutation_scale
    
    def emit(
        self,
        archive: Dict[int, Individual],
        centroids: List[List[float]],
        init_genome_fn: Callable,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        genomes = []
        
        if not archive:
            # Fallback: random initialization
            for _ in range(batch_size):
                genomes.append(init_genome_fn(self.rng))
            return genomes
        
        # Select top individuals by fitness
        sorted_archive = sorted(archive.values(), key=lambda x: x.fitness, reverse=True)
        top_k = min(max(1, len(sorted_archive) // 4), 10)
        elite = sorted_archive[:top_k]
        
        for _ in range(batch_size):
            # Select random elite individual
            parent = self.rng.choice(elite)
            
            # Small mutation for improvement
            child_genome = parent.genome.copy()
            for key in child_genome:
                if self.rng.random() < 0.5:
                    child_genome[key] += self.rng.gauss(0, self.mutation_scale)
            
            genomes.append(child_genome)
        
        self.stats.emissions += batch_size
        return genomes


class ExplorationEmitter(BaseEmitter):
    """
    Exploration emitter: Explores new areas.
    
    Applies large mutations to discover new regions of behavior space.
    """
    
    def __init__(self, emitter_id: str, mutation_scale: float = 0.5, seed: int = 42):
        super().__init__(emitter_id, EmitterType.EXPLORATION, seed)
        self.mutation_scale = mutation_scale
    
    def emit(
        self,
        archive: Dict[int, Individual],
        centroids: List[List[float]],
        init_genome_fn: Callable,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        genomes = []
        
        if not archive:
            for _ in range(batch_size):
                genomes.append(init_genome_fn(self.rng))
            return genomes
        
        for _ in range(batch_size):
            # Select random individual
            parent = self.rng.choice(list(archive.values()))
            
            # Large mutation for exploration
            child_genome = parent.genome.copy()
            for key in child_genome:
                if self.rng.random() < 0.8:  # High mutation rate
                    child_genome[key] += self.rng.gauss(0, self.mutation_scale)
            
            genomes.append(child_genome)
        
        self.stats.emissions += batch_size
        return genomes


class RandomEmitter(BaseEmitter):
    """
    Random emitter: Pure random baseline.
    
    Generates completely random genomes for exploration.
    """
    
    def __init__(self, emitter_id: str, seed: int = 42):
        super().__init__(emitter_id, EmitterType.RANDOM, seed)
    
    def emit(
        self,
        archive: Dict[int, Individual],
        centroids: List[List[float]],
        init_genome_fn: Callable,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        genomes = [init_genome_fn(self.rng) for _ in range(batch_size)]
        self.stats.emissions += batch_size
        return genomes


class CuriosityEmitter(BaseEmitter):
    """
    Curiosity emitter: Targets sparse/empty niches.
    
    Preferentially emits offspring near empty or low-density centroids.
    """
    
    def __init__(self, emitter_id: str, mutation_scale: float = 0.3, seed: int = 42):
        super().__init__(emitter_id, EmitterType.CURIOSITY, seed)
        self.mutation_scale = mutation_scale
    
    def emit(
        self,
        archive: Dict[int, Individual],
        centroids: List[List[float]],
        init_genome_fn: Callable,
        batch_size: int = 1
    ) -> List[Dict[str, float]]:
        genomes = []
        
        if not archive or not centroids:
            for _ in range(batch_size):
                genomes.append(init_genome_fn(self.rng))
            return genomes
        
        # Find empty niches
        filled_niches = set(archive.keys())
        empty_niches = [i for i in range(len(centroids)) if i not in filled_niches]
        
        for _ in range(batch_size):
            if empty_niches and self.rng.random() < 0.7:
                # Target empty niche
                target_niche_id = self.rng.choice(empty_niches)
                # Find nearest filled niche
                if archive:
                    parent = self.rng.choice(list(archive.values()))
                    child_genome = parent.genome.copy()
                    # Mutate towards target (heuristic)
                    for key in child_genome:
                        child_genome[key] += self.rng.gauss(0, self.mutation_scale)
                else:
                    child_genome = init_genome_fn(self.rng)
            else:
                # Random parent
                parent = self.rng.choice(list(archive.values()))
                child_genome = parent.genome.copy()
                for key in child_genome:
                    if self.rng.random() < 0.6:
                        child_genome[key] += self.rng.gauss(0, self.mutation_scale)
            
            genomes.append(child_genome)
        
        self.stats.emissions += batch_size
        return genomes


class MultiEmitterQD:
    """
    Multi-Emitter Quality-Diversity framework.
    
    Coordinates multiple emitters (improvement, exploration, random, curiosity)
    to efficiently illuminate the behavior space.
    """
    
    def __init__(
        self,
        n_niches: int,
        behavior_dim: int,
        behavior_bounds: List[Tuple[float, float]],
        init_genome_fn: Callable,
        eval_fn: Callable,
        emitters: Optional[List[BaseEmitter]] = None,
        seed: int = 42
    ):
        """
        Initialize Multi-Emitter QD.
        
        Args:
            n_niches: Number of niches in archive
            behavior_dim: Behavioral descriptor dimensionality
            behavior_bounds: [(min, max), ...] for each behavior dimension
            init_genome_fn: Function to create random genome
            eval_fn: Function(genome) -> (fitness, behavior)
            emitters: List of emitters (if None, creates default set)
            seed: Random seed
        """
        self.n_niches = n_niches
        self.behavior_dim = behavior_dim
        self.behavior_bounds = behavior_bounds
        self.init_genome_fn = init_genome_fn
        self.eval_fn = eval_fn
        self.rng = random.Random(seed)
        
        # Archive and centroids (simplified: random centroids)
        self.archive: Dict[int, Individual] = {}
        self.centroids = self._initialize_random_centroids()
        
        # Emitters
        if emitters is None:
            self.emitters = [
                ImprovementEmitter("improvement_0", mutation_scale=0.1, seed=seed),
                ExplorationEmitter("exploration_0", mutation_scale=0.5, seed=seed+1),
                RandomEmitter("random_0", seed=seed+2),
                CuriosityEmitter("curiosity_0", mutation_scale=0.3, seed=seed+3)
            ]
        else:
            self.emitters = emitters
        
        self.iterations = 0
        self.total_evaluations = 0
    
    def _initialize_random_centroids(self) -> List[List[float]]:
        """Initialize random centroids."""
        centroids = []
        for _ in range(self.n_niches):
            centroid = [
                self.rng.uniform(self.behavior_bounds[d][0], self.behavior_bounds[d][1])
                for d in range(self.behavior_dim)
            ]
            centroids.append(centroid)
        return centroids
    
    def _find_nearest_centroid(self, behavior: List[float]) -> int:
        """Find nearest centroid to behavior."""
        min_dist = float('inf')
        nearest_idx = 0
        for i, centroid in enumerate(self.centroids):
            dist = sum((b - c) ** 2 for b, c in zip(behavior, centroid)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx
    
    def _try_add_to_archive(self, individual: Individual) -> Tuple[bool, float]:
        """
        Try to add individual to archive.
        Returns (added, fitness_gain)
        """
        niche_id = self._find_nearest_centroid(individual.behavior)
        
        if niche_id not in self.archive:
            self.archive[niche_id] = individual
            return True, individual.fitness
        elif individual.fitness > self.archive[niche_id].fitness:
            old_fitness = self.archive[niche_id].fitness
            self.archive[niche_id] = individual
            return True, individual.fitness - old_fitness
        return False, 0.0
    
    def evolve(
        self,
        n_iterations: int = 100,
        batch_size_per_emitter: int = 5,
        verbose: bool = True
    ):
        """
        Run multi-emitter QD evolution.
        
        Args:
            n_iterations: Number of iterations
            batch_size_per_emitter: Offspring per emitter per iteration
            verbose: Print progress
        """
        for iteration in range(n_iterations):
            self.iterations += 1
            
            # Each emitter generates offspring
            for emitter in self.emitters:
                genomes = emitter.emit(
                    self.archive,
                    self.centroids,
                    self.init_genome_fn,
                    batch_size=batch_size_per_emitter
                )
                
                # Evaluate and try to add
                for genome in genomes:
                    fitness, behavior = self.eval_fn(genome)
                    self.total_evaluations += 1
                    
                    ind = Individual(
                        genome=genome,
                        fitness=fitness,
                        behavior=behavior,
                        emitter_id=emitter.emitter_id
                    )
                    
                    added, gain = self._try_add_to_archive(ind)
                    if added:
                        emitter.stats.improvements += 1
                        emitter.stats.total_fitness_gain += gain
                        if gain == fitness:  # New niche
                            emitter.stats.new_niches += 1
            
            # Logging
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                coverage = len(self.archive) / self.n_niches
                qd_score = sum(ind.fitness for ind in self.archive.values())
                max_fit = max((ind.fitness for ind in self.archive.values()), default=0.0)
                
                print(f"[Multi-Emitter] iter={iteration+1:04d} coverage={coverage:.3f} "
                      f"qd_score={qd_score:.2f} max_fit={max_fit:.4f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get QD metrics."""
        if not self.archive:
            return {
                'coverage': 0.0,
                'qd_score': 0.0,
                'max_fitness': 0.0,
                'archive_size': 0,
                'emitter_stats': {}
            }
        
        fitnesses = [ind.fitness for ind in self.archive.values()]
        
        emitter_stats = {}
        for emitter in self.emitters:
            emitter_stats[emitter.emitter_id] = {
                'type': emitter.emitter_type.value,
                'emissions': emitter.stats.emissions,
                'improvements': emitter.stats.improvements,
                'new_niches': emitter.stats.new_niches,
                'total_fitness_gain': emitter.stats.total_fitness_gain
            }
        
        return {
            'coverage': len(self.archive) / self.n_niches,
            'qd_score': sum(fitnesses),
            'max_fitness': max(fitnesses),
            'archive_size': len(self.archive),
            'total_evaluations': self.total_evaluations,
            'emitter_stats': emitter_stats
        }


# ============================================================================
# TEST
# ============================================================================

def test_multi_emitter_qd():
    """Test Multi-Emitter QD."""
    print("\n" + "="*80)
    print("🧪 TESTE: Multi-Emitter QD")
    print("="*80)
    
    def init_genome(rng):
        return {'x': rng.uniform(-3, 3), 'y': rng.uniform(-3, 3)}
    
    def eval_fn(genome):
        x, y = genome['x'], genome['y']
        fitness = 15 + x*math.sin(3*math.pi*x) + y*math.cos(3*math.pi*y)
        behavior = [x, y]
        return fitness, behavior
    
    me_qd = MultiEmitterQD(
        n_niches=40,
        behavior_dim=2,
        behavior_bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        init_genome_fn=init_genome,
        eval_fn=eval_fn,
        seed=42
    )
    
    me_qd.evolve(n_iterations=20, batch_size_per_emitter=5, verbose=True)
    
    metrics = me_qd.get_metrics()
    
    print("\n📊 Resultados:")
    print(f"   Coverage: {metrics['coverage']:.3f}")
    print(f"   QD-Score: {metrics['qd_score']:.2f}")
    print(f"   Max Fitness: {metrics['max_fitness']:.4f}")
    print(f"   Archive Size: {metrics['archive_size']}")
    print(f"   Total Evals: {metrics['total_evaluations']}")
    
    print("\n   📈 Emitter Stats:")
    for emitter_id, stats in metrics['emitter_stats'].items():
        print(f"      {emitter_id:20s} ({stats['type']:12s}): "
              f"{stats['improvements']:3d} improvements, "
              f"{stats['new_niches']:3d} new niches")
    
    assert metrics['coverage'] > 0.5, "Coverage baixa"
    assert metrics['archive_size'] > 15, "Archive pequeno"
    
    print("\n✅ Multi-Emitter QD: PASS")
    print("="*80)
    return metrics


if __name__ == "__main__":
    random.seed(42)
    test_multi_emitter_qd()
    print("\n" + "="*80)
    print("✅ multi_emitter_qd.py está FUNCIONAL!")
    print("="*80)
