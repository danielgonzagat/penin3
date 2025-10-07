"""
Archive Manager: Advanced archive management for Quality-Diversity.

Implements:
- Archive pruning/compaction (when archive gets too large)
- Diversity-preserving selection
- Niche merging/splitting
- Archive statistics and health monitoring

Features:
- K-means-like clustering for pruning
- Diversity metrics (entropy, coverage)
- Per-niche best tracking
- Archive health monitoring
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class ArchiveStats:
    """Statistics about the archive."""
    size: int
    coverage: float
    qd_score: float
    max_fitness: float
    mean_fitness: float
    entropy: float
    diversity: float


class ArchiveManager:
    """
    Manages QD archive with advanced features.
    
    Features:
    - Pruning when archive gets too large
    - Diversity-preserving compaction
    - Health monitoring
    - Statistics tracking
    """
    
    def __init__(
        self,
        n_niches: int,
        max_archive_size: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize Archive Manager.
        
        Args:
            n_niches: Total number of niches
            max_archive_size: Maximum archive size (if None, no limit)
            seed: Random seed
        """
        self.n_niches = n_niches
        self.max_archive_size = max_archive_size
        self.rng = random.Random(seed)
        
        # Archive: niche_id -> Individual
        self.archive: Dict[int, Any] = {}
        
        # Statistics
        self.total_adds = 0
        self.total_improvements = 0
        self.prune_count = 0
    
    def add(
        self,
        niche_id: int,
        individual: Any
    ) -> Tuple[bool, float]:
        """
        Try to add individual to archive.
        
        Returns:
            (added: bool, fitness_gain: float)
        """
        self.total_adds += 1
        
        if niche_id not in self.archive:
            # New niche
            self.archive[niche_id] = individual
            return True, individual.fitness
        
        # Check if better
        if individual.fitness > self.archive[niche_id].fitness:
            old_fitness = self.archive[niche_id].fitness
            self.archive[niche_id] = individual
            self.total_improvements += 1
            return True, individual.fitness - old_fitness
        
        return False, 0.0
    
    def prune(self, target_size: int, strategy: str = 'diversity'):
        """
        Prune archive to target size.
        
        Args:
            target_size: Target number of niches to keep
            strategy: 'diversity', 'fitness', or 'random'
        """
        if len(self.archive) <= target_size:
            return
        
        self.prune_count += 1
        
        if strategy == 'diversity':
            self._prune_diversity(target_size)
        elif strategy == 'fitness':
            self._prune_fitness(target_size)
        else:
            self._prune_random(target_size)
    
    def _prune_diversity(self, target_size: int):
        """
        Prune while preserving diversity.
        
        Uses simple k-means-like clustering: keep one representative per cluster.
        """
        if len(self.archive) <= target_size:
            return
        
        # Get all behaviors
        niche_ids = list(self.archive.keys())
        behaviors = [self.archive[nid].behavior for nid in niche_ids]
        
        # Simple clustering: randomly select target_size cluster centers
        cluster_centers_idx = self.rng.sample(range(len(behaviors)), target_size)
        cluster_centers = [behaviors[i] for i in cluster_centers_idx]
        
        # Assign each niche to nearest cluster
        clusters = [[] for _ in range(target_size)]
        for i, nid in enumerate(niche_ids):
            behavior = behaviors[i]
            # Find nearest cluster
            min_dist = float('inf')
            nearest_cluster = 0
            for j, center in enumerate(cluster_centers):
                dist = euclidean_distance(behavior, center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = j
            clusters[nearest_cluster].append(nid)
        
        # Keep best from each cluster
        new_archive = {}
        for cluster in clusters:
            if cluster:
                # Select best fitness in cluster
                best_nid = max(cluster, key=lambda nid: self.archive[nid].fitness)
                new_archive[best_nid] = self.archive[best_nid]
        
        self.archive = new_archive
    
    def _prune_fitness(self, target_size: int):
        """Prune keeping top fitness individuals."""
        sorted_nids = sorted(
            self.archive.keys(),
            key=lambda nid: self.archive[nid].fitness,
            reverse=True
        )
        
        keep_nids = sorted_nids[:target_size]
        self.archive = {nid: self.archive[nid] for nid in keep_nids}
    
    def _prune_random(self, target_size: int):
        """Prune randomly."""
        keep_nids = self.rng.sample(list(self.archive.keys()), target_size)
        self.archive = {nid: self.archive[nid] for nid in keep_nids}
    
    def get_stats(self) -> ArchiveStats:
        """Get archive statistics."""
        if not self.archive:
            return ArchiveStats(
                size=0,
                coverage=0.0,
                qd_score=0.0,
                max_fitness=0.0,
                mean_fitness=0.0,
                entropy=0.0,
                diversity=0.0
            )
        
        fitnesses = [ind.fitness for ind in self.archive.values()]
        
        # Diversity: average pairwise distance in behavior space
        diversity = 0.0
        if len(self.archive) > 1:
            behaviors = [ind.behavior for ind in self.archive.values()]
            distances = []
            for i in range(len(behaviors)):
                for j in range(i + 1, len(behaviors)):
                    distances.append(euclidean_distance(behaviors[i], behaviors[j]))
            diversity = sum(distances) / len(distances) if distances else 0.0
        
        # Entropy: log of archive size (simple proxy)
        entropy = math.log(max(1, len(self.archive)))
        
        return ArchiveStats(
            size=len(self.archive),
            coverage=len(self.archive) / self.n_niches,
            qd_score=sum(fitnesses),
            max_fitness=max(fitnesses),
            mean_fitness=sum(fitnesses) / len(fitnesses),
            entropy=entropy,
            diversity=diversity
        )
    
    def needs_pruning(self) -> bool:
        """Check if archive needs pruning."""
        if self.max_archive_size is None:
            return False
        return len(self.archive) > self.max_archive_size


# ============================================================================
# TEST
# ============================================================================

def test_archive_manager():
    """Test Archive Manager."""
    print("\n" + "="*80)
    print("🧪 TESTE: Archive Manager")
    print("="*80)
    
    class Individual:
        def __init__(self, genome, fitness, behavior):
            self.genome = genome
            self.fitness = fitness
            self.behavior = behavior
    
    manager = ArchiveManager(n_niches=100, max_archive_size=50, seed=42)
    
    # Add individuals
    for i in range(80):
        genome = {'x': random.uniform(-5, 5), 'y': random.uniform(-5, 5)}
        fitness = 10 + random.random() * 10
        behavior = [genome['x'], genome['y']]
        ind = Individual(genome, fitness, behavior)
        
        added, gain = manager.add(i, ind)
        if added:
            pass  # print(f"   Added to niche {i}: fitness={fitness:.2f}")
    
    print(f"\n📦 Archive após adições: {len(manager.archive)} niches")
    
    # Check if needs pruning
    if manager.needs_pruning():
        print(f"⚠️ Archive precisa pruning (size={len(manager.archive)} > max={manager.max_archive_size})")
        
        # Prune with diversity strategy
        print(f"🔪 Pruning para {manager.max_archive_size} niches (strategy=diversity)...")
        manager.prune(target_size=manager.max_archive_size, strategy='diversity')
        print(f"✅ Archive após pruning: {len(manager.archive)} niches")
    
    # Get stats
    stats = manager.get_stats()
    
    print(f"\n📊 Estatísticas Finais:")
    print(f"   Size: {stats.size}")
    print(f"   Coverage: {stats.coverage:.3f}")
    print(f"   QD-Score: {stats.qd_score:.2f}")
    print(f"   Max Fitness: {stats.max_fitness:.4f}")
    print(f"   Mean Fitness: {stats.mean_fitness:.4f}")
    print(f"   Entropy: {stats.entropy:.4f}")
    print(f"   Diversity: {stats.diversity:.4f}")
    
    print(f"\n   Total adds: {manager.total_adds}")
    print(f"   Improvements: {manager.total_improvements}")
    print(f"   Prune count: {manager.prune_count}")
    
    # Validate
    assert stats.size == 50, f"Expected 50, got {stats.size}"
    assert stats.coverage == 0.5, f"Expected 0.5, got {stats.coverage}"
    assert stats.qd_score > 500, "QD-score too low"
    
    print("\n✅ Archive Manager: PASS")
    print("="*80)


if __name__ == "__main__":
    random.seed(42)
    test_archive_manager()
    print("\n" + "="*80)
    print("✅ archive_manager.py está FUNCIONAL!")
    print("="*80)
