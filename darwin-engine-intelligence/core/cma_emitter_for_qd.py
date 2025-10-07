"""
CMA-ES Emitter for Quality-Diversity (CMA-MEGA style)

Implements a CMA-ES emitter that maintains separate CMA-ES instances per niche,
enabling efficient exploitation while maintaining diversity across the archive.

This is a key component for CMA-MEGA (Covariance Matrix Adaptation MAP-Elites
via a Gradient Arborescence), which combines the power of CMA-ES with the
diversity preservation of MAP-Elites.

References:
- Fontaine, M. C., et al. (2020). "Covariance Matrix Adaptation for the 
  Rapid Illumination of Behavior Space" (CMA-ME)
"""

import random
import math
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class CMAESState:
    """State of a CMA-ES instance."""
    mean: List[float]
    sigma: float
    generation: int = 0
    best_fitness: float = float('-inf')


class CMAESEmitter:
    """
    CMA-ES Emitter for QD.
    
    Maintains CMA-ES instances per niche or region, enabling local exploitation
    while the overall QD framework maintains global diversity.
    """
    
    def __init__(
        self,
        emitter_id: str,
        initial_sigma: float = 0.3,
        max_instances: int = 20,
        seed: int = 42
    ):
        """
        Initialize CMA-ES Emitter.
        
        Args:
            emitter_id: Unique identifier
            initial_sigma: Initial step size for CMA-ES
            max_instances: Maximum number of CMA-ES instances to maintain
            seed: Random seed
        """
        self.emitter_id = emitter_id
        self.initial_sigma = initial_sigma
        self.max_instances = max_instances
        self.rng = random.Random(seed)
        
        # CMA-ES instances per niche
        self.cma_instances: Dict[int, CMAESState] = {}
        
        # Statistics
        self.emissions = 0
        self.improvements = 0
        self.new_niches = 0
    
    def emit(
        self,
        archive: Dict[int, Any],
        batch_size: int = 5,
        genome_keys: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Emit offspring using CMA-ES.
        
        Args:
            archive: Current QD archive (niche_id -> Individual)
            batch_size: Number of offspring to generate
            genome_keys: Keys for genome dict (if None, inferred from archive)
        
        Returns:
            List of genomes
        """
        genomes = []
        
        if not archive:
            # No archive yet, return empty
            return genomes
        
        # Infer genome keys if not provided
        if genome_keys is None:
            first_ind = next(iter(archive.values()))
            genome_keys = list(first_ind.genome.keys())
        
        # Prune CMA instances if too many
        if len(self.cma_instances) > self.max_instances:
            # Keep only the most recent or best performing
            niche_ids = list(self.cma_instances.keys())
            to_remove = self.rng.sample(niche_ids, len(niche_ids) - self.max_instances)
            for nid in to_remove:
                del self.cma_instances[nid]
        
        # Select random niches to sample from (with replacement for batch_size)
        filled_niches = list(archive.keys())
        selected_niches = self.rng.choices(filled_niches, k=batch_size)
        
        for niche_id in selected_niches:
            individual = archive[niche_id]
            
            # Initialize CMA-ES for this niche if not exists
            if niche_id not in self.cma_instances:
                # Initialize mean from individual's genome
                mean = [individual.genome[k] for k in genome_keys]
                self.cma_instances[niche_id] = CMAESState(
                    mean=mean,
                    sigma=self.initial_sigma,
                    best_fitness=individual.fitness
                )
            
            cma = self.cma_instances[niche_id]
            
            # Sample from CMA-ES distribution
            offspring_values = self._sample_from_cma(cma, genome_keys)
            offspring_genome = {k: v for k, v in zip(genome_keys, offspring_values)}
            
            genomes.append(offspring_genome)
            
            # Update CMA generation counter
            cma.generation += 1
        
        self.emissions += len(genomes)
        return genomes
    
    def _sample_from_cma(
        self,
        cma: CMAESState,
        genome_keys: List[str]
    ) -> List[float]:
        """
        Sample a single individual from CMA-ES distribution.
        
        Simple version: N(mean, sigma^2 * I)
        (Full CMA-ES would use covariance matrix)
        """
        n_dims = len(genome_keys)
        sample = []
        
        for i in range(n_dims):
            # Gaussian perturbation around mean
            value = cma.mean[i] + self.rng.gauss(0, cma.sigma)
            sample.append(value)
        
        return sample
    
    def update(
        self,
        niche_id: int,
        genome: Dict[str, float],
        fitness: float,
        improved: bool
    ):
        """
        Update CMA-ES state for a niche.
        
        Args:
            niche_id: Niche that was updated
            genome: Genome that was evaluated
            fitness: Fitness achieved
            improved: Whether this improved the niche
        """
        if niche_id not in self.cma_instances:
            return
        
        cma = self.cma_instances[niche_id]
        
        if improved:
            self.improvements += 1
            
            # Update CMA mean towards better solution
            genome_values = list(genome.values())
            learning_rate = 0.3  # Simple update
            
            for i in range(len(cma.mean)):
                if i < len(genome_values):
                    cma.mean[i] = (1 - learning_rate) * cma.mean[i] + learning_rate * genome_values[i]
            
            # Slightly reduce sigma (exploitation)
            cma.sigma *= 0.95
            cma.sigma = max(0.01, cma.sigma)  # Lower bound
            
            cma.best_fitness = fitness
        else:
            # No improvement: slightly increase sigma (exploration)
            cma.sigma *= 1.02
            cma.sigma = min(1.0, cma.sigma)  # Upper bound
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            'emitter_id': self.emitter_id,
            'emissions': self.emissions,
            'improvements': self.improvements,
            'new_niches': self.new_niches,
            'active_cma_instances': len(self.cma_instances),
            'avg_sigma': sum(c.sigma for c in self.cma_instances.values()) / max(1, len(self.cma_instances))
        }


# ============================================================================
# TEST
# ============================================================================

def test_cma_emitter():
    """Test CMA-ES Emitter."""
    print("\n" + "="*80)
    print("🧪 TESTE: CMA-ES Emitter for QD")
    print("="*80)
    
    # Mock Individual class
    class Individual:
        def __init__(self, genome, fitness):
            self.genome = genome
            self.fitness = fitness
            self.behavior = list(genome.values())
    
    # Create emitter
    emitter = CMAESEmitter(
        emitter_id="cma_emitter_0",
        initial_sigma=0.3,
        max_instances=10,
        seed=42
    )
    
    # Mock archive
    archive = {}
    for i in range(5):
        genome = {'x': random.uniform(-2, 2), 'y': random.uniform(-2, 2)}
        fitness = 10 + sum(genome.values())
        archive[i] = Individual(genome, fitness)
    
    print(f"📦 Archive inicial: {len(archive)} niches")
    
    # Emit offspring
    genomes = emitter.emit(archive, batch_size=10, genome_keys=['x', 'y'])
    
    print(f"🧬 Emitted: {len(genomes)} genomes")
    
    # Simulate evaluation and updates
    for i, genome in enumerate(genomes):
        niche_id = i % len(archive)
        fitness = 10 + sum(genome.values())
        improved = fitness > archive[niche_id].fitness
        
        emitter.update(niche_id, genome, fitness, improved)
    
    # Get stats
    stats = emitter.get_stats()
    
    print(f"\n📊 Estatísticas:")
    print(f"   Emissions: {stats['emissions']}")
    print(f"   Improvements: {stats['improvements']}")
    print(f"   Active CMA instances: {stats['active_cma_instances']}")
    print(f"   Avg sigma: {stats['avg_sigma']:.4f}")
    
    # Validate
    assert stats['emissions'] == 10, "Wrong emission count"
    assert stats['active_cma_instances'] <= 10, "Too many CMA instances"
    assert len(genomes) == 10, "Wrong number of genomes"
    
    print("\n✅ CMA-ES Emitter: PASS")
    print("="*80)
    return stats


if __name__ == "__main__":
    random.seed(42)
    test_cma_emitter()
    print("\n" + "="*80)
    print("✅ cma_emitter_for_qd.py está FUNCIONAL!")
    print("="*80)
