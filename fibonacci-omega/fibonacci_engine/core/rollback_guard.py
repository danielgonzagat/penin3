"""
Rollback Guard: Automatic Regression Detection and Protection

Monitors performance and prevents destructive changes by detecting
regressions and triggering rollbacks when necessary.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fibonacci_engine.core.map_elites import MAPElites, Candidate


@dataclass
class PerformanceSnapshot:
    """
    A snapshot of performance metrics at a specific point in time.
    
    Attributes:
        generation: Generation number.
        best_fitness: Best overall fitness.
        mean_fitness: Mean fitness of elites.
        median_fitness: Median fitness of elites.
        coverage: Archive coverage.
        n_elites: Number of elites.
    """
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    coverage: float
    n_elites: int
    
    def to_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "median_fitness": self.median_fitness,
            "coverage": self.coverage,
            "n_elites": self.n_elites,
        }


class RollbackGuard:
    """
    Monitors performance and detects regressions.
    
    A regression is detected if:
    - Best fitness drops significantly
    - Mean fitness of elites drops significantly
    - Coverage decreases dramatically
    
    Args:
        rollback_delta: Maximum allowed drop in mean fitness (0.0 to 1.0).
        window_size: Number of generations to track for baseline.
        min_generations: Minimum generations before enabling rollback checks.
    """
    
    def __init__(
        self,
        rollback_delta: float = 0.02,
        window_size: int = 5,
        min_generations: int = 10,
    ):
        self.rollback_delta = rollback_delta
        self.window_size = window_size
        self.min_generations = min_generations
        
        self.history: List[PerformanceSnapshot] = []
        self.baseline_best: Optional[float] = None
        self.baseline_mean: Optional[float] = None
        
        self.n_rollbacks_triggered = 0
        self.last_rollback_generation = -1
    
    def record(self, generation: int, archive: MAPElites):
        """
        Record current performance.
        
        Args:
            generation: Current generation number.
            archive: Current MAP-Elites archive.
        """
        stats = archive.get_statistics()
        
        snapshot = PerformanceSnapshot(
            generation=generation,
            best_fitness=stats.get("best_fitness", 0.0) or 0.0,
            mean_fitness=stats.get("mean_fitness", 0.0) or 0.0,
            median_fitness=stats.get("median_fitness", 0.0) or 0.0,
            coverage=stats.get("coverage", 0.0),
            n_elites=stats.get("n_elites", 0),
        )
        
        self.history.append(snapshot)
        
        # Update baseline using windowed average
        self._update_baseline()
    
    def _update_baseline(self):
        """Update baseline performance from recent history."""
        if len(self.history) < self.min_generations:
            # Not enough data yet
            return
        
        # Use last window_size snapshots for baseline
        recent = self.history[-self.window_size:]
        
        self.baseline_best = max(s.best_fitness for s in recent)
        self.baseline_mean = np.mean([s.mean_fitness for s in recent])
    
    def check_regression(self, generation: int, archive: MAPElites) -> bool:
        """
        Check if current performance represents a regression.
        
        Args:
            generation: Current generation number.
            archive: Current MAP-Elites archive.
            
        Returns:
            True if regression detected, False otherwise.
        """
        if len(self.history) < self.min_generations:
            # Not enough data to determine regression
            return False
        
        if self.baseline_best is None or self.baseline_mean is None:
            return False
        
        stats = archive.get_statistics()
        current_best = stats.get("best_fitness", 0.0) or 0.0
        current_mean = stats.get("mean_fitness", 0.0) or 0.0
        
        # Check for significant drops
        best_drop = self.baseline_best - current_best
        mean_drop = self.baseline_mean - current_mean
        
        # Regression if mean fitness drops by more than rollback_delta
        if mean_drop > self.rollback_delta:
            self.n_rollbacks_triggered += 1
            self.last_rollback_generation = generation
            return True
        
        # Also check for catastrophic best fitness drop
        if best_drop > self.rollback_delta * 2:
            self.n_rollbacks_triggered += 1
            self.last_rollback_generation = generation
            return True
        
        return False
    
    def should_rollback(self, generation: int, archive: MAPElites) -> Dict[str, Any]:
        """
        Check if rollback should be triggered.
        
        Args:
            generation: Current generation number.
            archive: Current MAP-Elites archive.
            
        Returns:
            Dictionary with rollback decision and details.
        """
        is_regression = self.check_regression(generation, archive)
        
        stats = archive.get_statistics()
        
        return {
            "should_rollback": is_regression,
            "generation": generation,
            "current_best": stats.get("best_fitness", 0.0),
            "baseline_best": self.baseline_best,
            "current_mean": stats.get("mean_fitness", 0.0),
            "baseline_mean": self.baseline_mean,
            "rollback_delta": self.rollback_delta,
            "n_rollbacks_triggered": self.n_rollbacks_triggered,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rollback guard statistics.
        
        Returns:
            Dictionary with statistics.
        """
        if not self.history:
            return {
                "n_generations": 0,
                "baseline_best": None,
                "baseline_mean": None,
                "n_rollbacks_triggered": 0,
            }
        
        return {
            "n_generations": len(self.history),
            "baseline_best": self.baseline_best,
            "baseline_mean": self.baseline_mean,
            "n_rollbacks_triggered": self.n_rollbacks_triggered,
            "last_rollback_generation": self.last_rollback_generation,
            "current_best": self.history[-1].best_fitness,
            "current_mean": self.history[-1].mean_fitness,
        }
    
    def get_history(self) -> List[Dict]:
        """Get performance history."""
        return [s.to_dict() for s in self.history]
    
    def reset(self):
        """Reset the rollback guard."""
        self.history.clear()
        self.baseline_best = None
        self.baseline_mean = None
        self.n_rollbacks_triggered = 0
        self.last_rollback_generation = -1
