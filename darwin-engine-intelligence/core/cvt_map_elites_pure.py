"""
CVT-MAP-Elites: Centroidal Voronoi Tessellation MAP-Elites
Pure Python implementation using Lloyd's algorithm for CVT.

This implements a more sophisticated version of MAP-Elites that uses
Centroidal Voronoi Tessellation to create a better coverage of the
behavioral space, especially in high dimensions.

References:
- Vassiliades, V., Chatzilygeroudis, K., & Mouret, J. B. (2018).
  "Using centroidal Voronoi tessellations to scale up the multi-dimensional 
  archive of phenotypic elites algorithm"
"""

import random
import math
import time
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Individual:
    """Represents an individual in the population."""
    genome: Dict[str, float]
    fitness: float = 0.0
    behavior: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class CVTMAPElites:
    """
    CVT-MAP-Elites: Quality-Diversity algorithm using Centroidal Voronoi Tessellation.
    
    Uses Lloyd's algorithm to compute centroids that better cover the behavioral space.
    This is especially useful for high-dimensional behavioral descriptors.
    """
    
    def __init__(
        self,
        n_niches: int,
        behavior_dim: int,
        behavior_bounds: List[Tuple[float, float]],
        init_genome_fn: Callable,
        eval_fn: Callable,
        mutation_fn: Callable,
        crossover_fn: Optional[Callable] = None,
        seed: int = 42
    ):
        """
        Initialize CVT-MAP-Elites.
        
        Args:
            n_niches: Number of niches (centroids) in the archive
            behavior_dim: Dimensionality of behavioral descriptor
            behavior_bounds: [(min, max), ...] for each behavior dimension
            init_genome_fn: Function to create random genome
            eval_fn: Function(genome) -> (fitness, behavior)
            mutation_fn: Function(genome) -> mutated_genome
            crossover_fn: Optional function(genome1, genome2) -> child_genome
            seed: Random seed
        """
        if len(behavior_bounds) != behavior_dim:
            raise ValueError(f"behavior_bounds length {len(behavior_bounds)} != behavior_dim {behavior_dim}")
        
        self.n_niches = n_niches
        self.behavior_dim = behavior_dim
        self.behavior_bounds = behavior_bounds
        self.init_genome_fn = init_genome_fn
        self.eval_fn = eval_fn
        self.mutation_fn = mutation_fn
        self.crossover_fn = crossover_fn
        self.rng = random.Random(seed)
        
        # Archive: dict[niche_id -> Individual]
        self.archive: Dict[int, Individual] = {}
        
        # Centroids computed via Lloyd's algorithm
        self.centroids: List[List[float]] = []
        
        # Statistics
        self.iterations = 0
        self.total_evaluations = 0
        
    def _initialize_centroids_lloyd(self, n_iterations: int = 20, n_samples: int = 10000):
        """
        Initialize centroids using Lloyd's algorithm for CVT.
        
        This creates a more uniform distribution of centroids in the behavior space
        compared to random initialization.
        """
        # Step 1: Generate random samples in behavior space
        samples = []
        for _ in range(n_samples):
            sample = [
                self.rng.uniform(self.behavior_bounds[d][0], self.behavior_bounds[d][1])
                for d in range(self.behavior_dim)
            ]
            samples.append(sample)
        
        # Step 2: Initialize centroids randomly from samples
        self.centroids = self.rng.sample(samples, self.n_niches)
        
        # Step 3: Lloyd's algorithm iterations
        for iteration in range(n_iterations):
            # Assign each sample to nearest centroid
            assignments = [[] for _ in range(self.n_niches)]
            for sample in samples:
                nearest_idx = self._find_nearest_centroid(sample)
                assignments[nearest_idx].append(sample)
            
            # Recompute centroids as mean of assigned samples
            for i in range(self.n_niches):
                if len(assignments[i]) > 0:
                    # Compute mean for each dimension
                    new_centroid = []
                    for d in range(self.behavior_dim):
                        mean_d = sum(s[d] for s in assignments[i]) / len(assignments[i])
                        new_centroid.append(mean_d)
                    self.centroids[i] = new_centroid
        
        # Ensure centroids are within bounds
        for i in range(self.n_niches):
            for d in range(self.behavior_dim):
                self.centroids[i][d] = max(
                    self.behavior_bounds[d][0],
                    min(self.behavior_bounds[d][1], self.centroids[i][d])
                )
    
    def _find_nearest_centroid(self, behavior: List[float]) -> int:
        """Find the index of the nearest centroid to the given behavior."""
        if not self.centroids:
            raise ValueError("Centroids not initialized. Call initialize() first.")
        
        min_dist = float('inf')
        nearest_idx = 0
        for i, centroid in enumerate(self.centroids):
            dist = euclidean_distance(behavior, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx
    
    def initialize(self, initial_population_size: int = 100):
        """
        Initialize the archive with random individuals and CVT centroids.
        """
        # Initialize centroids using Lloyd's algorithm
        self._initialize_centroids_lloyd()
        
        # Create initial random population
        for _ in range(initial_population_size):
            genome = self.init_genome_fn(self.rng)
            fitness, behavior = self.eval_fn(genome)
            self.total_evaluations += 1
            
            ind = Individual(genome=genome, fitness=fitness, behavior=behavior)
            self._try_add_to_archive(ind)
    
    def _try_add_to_archive(self, individual: Individual) -> bool:
        """
        Try to add individual to archive.
        Returns True if added (either new niche or better fitness).
        """
        # Find nearest centroid (niche)
        niche_id = self._find_nearest_centroid(individual.behavior)
        
        # Check if niche is empty or if individual is better
        if niche_id not in self.archive:
            self.archive[niche_id] = individual
            return True
        elif individual.fitness > self.archive[niche_id].fitness:
            self.archive[niche_id] = individual
            return True
        return False
    
    def evolve(
        self,
        n_iterations: int = 1000,
        batch_size: int = 20,
        mutation_rate: float = 0.8,
        verbose: bool = True
    ):
        """
        Run CVT-MAP-Elites evolution.
        
        Args:
            n_iterations: Number of evolution iterations
            batch_size: Number of offspring per iteration
            mutation_rate: Probability of mutation vs crossover
            verbose: Print progress
        """
        if not self.centroids:
            self.initialize()
        
        for iteration in range(n_iterations):
            self.iterations += 1
            
            # Generate batch of offspring
            for _ in range(batch_size):
                # Select random parent from archive
                if not self.archive:
                    # Fallback: create random individual
                    genome = self.init_genome_fn(self.rng)
                else:
                    parent = self.rng.choice(list(self.archive.values()))
                    
                    # Mutation or crossover
                    if self.crossover_fn and self.rng.random() > mutation_rate and len(self.archive) > 1:
                        # Crossover
                        parent2 = self.rng.choice(list(self.archive.values()))
                        genome = self.crossover_fn(parent.genome, parent2.genome)
                    else:
                        # Mutation
                        genome = self.mutation_fn(parent.genome)
                
                # Evaluate offspring
                fitness, behavior = self.eval_fn(genome)
                self.total_evaluations += 1
                
                offspring = Individual(genome=genome, fitness=fitness, behavior=behavior)
                self._try_add_to_archive(offspring)
            
            # Logging
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                coverage = len(self.archive) / self.n_niches
                max_fit = max(ind.fitness for ind in self.archive.values()) if self.archive else 0.0
                mean_fit = sum(ind.fitness for ind in self.archive.values()) / max(1, len(self.archive))
                qd_score = sum(ind.fitness for ind in self.archive.values())
                
                print(f"[CVT-ME] iter={iteration+1:04d} coverage={coverage:.3f} "
                      f"max_fit={max_fit:.4f} mean_fit={mean_fit:.4f} qd_score={qd_score:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get quality-diversity metrics."""
        if not self.archive:
            return {
                'coverage': 0.0,
                'qd_score': 0.0,
                'max_fitness': 0.0,
                'mean_fitness': 0.0,
                'archive_size': 0,
                'total_evaluations': self.total_evaluations
            }
        
        fitnesses = [ind.fitness for ind in self.archive.values()]
        
        return {
            'coverage': len(self.archive) / self.n_niches,
            'qd_score': sum(fitnesses),
            'max_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'archive_size': len(self.archive),
            'total_evaluations': self.total_evaluations
        }


# ============================================================================
# TEST & DEMO
# ============================================================================

def test_cvt_map_elites():
    """Test CVT-MAP-Elites on a simple 2D problem."""
    print("\n" + "="*80)
    print("🧪 TESTE: CVT-MAP-Elites")
    print("="*80)
    
    # Simple 2D problem: maximize x*sin(4*pi*x) + y*cos(4*pi*y)
    def init_genome(rng):
        return {'x': rng.uniform(-2, 2), 'y': rng.uniform(-2, 2)}
    
    def eval_fn(genome):
        x, y = genome['x'], genome['y']
        # Fitness: rastrigin-like function
        fitness = 10 + x*math.sin(4*math.pi*x) + y*math.cos(4*math.pi*y)
        # Behavior: just (x, y) position
        behavior = [x, y]
        return fitness, behavior
    
    def mutate(genome):
        g = genome.copy()
        if random.random() < 0.5:
            g['x'] += random.gauss(0, 0.3)
            g['x'] = max(-2, min(2, g['x']))
        if random.random() < 0.5:
            g['y'] += random.gauss(0, 0.3)
            g['y'] = max(-2, min(2, g['y']))
        return g
    
    # Create CVT-MAP-Elites with 50 niches
    cvt_me = CVTMAPElites(
        n_niches=50,
        behavior_dim=2,
        behavior_bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        init_genome_fn=init_genome,
        eval_fn=eval_fn,
        mutation_fn=mutate,
        seed=42
    )
    
    # Initialize and evolve
    cvt_me.initialize(initial_population_size=100)
    cvt_me.evolve(n_iterations=20, batch_size=10, verbose=True)
    
    # Get metrics
    metrics = cvt_me.get_metrics()
    
    print("\n📊 Resultados Finais:")
    print(f"   Coverage: {metrics['coverage']:.3f} ({metrics['archive_size']}/{cvt_me.n_niches} nichos)")
    print(f"   QD-Score: {metrics['qd_score']:.2f}")
    print(f"   Max Fitness: {metrics['max_fitness']:.4f}")
    print(f"   Mean Fitness: {metrics['mean_fitness']:.4f}")
    print(f"   Total Evaluations: {metrics['total_evaluations']}")
    
    # Validate
    assert metrics['coverage'] > 0.3, "Coverage muito baixa"
    assert metrics['archive_size'] > 10, "Archive muito pequeno"
    assert metrics['qd_score'] > 100, "QD-score muito baixo"
    
    print("\n✅ CVT-MAP-Elites: PASS")
    print("="*80)
    return metrics


if __name__ == "__main__":
    random.seed(42)
    test_cvt_map_elites()
    print("\n" + "="*80)
    print("✅ cvt_map_elites_pure.py está FUNCIONAL!")
    print("="*80)
