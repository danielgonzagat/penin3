"""
MAP-Elites: Quality-Diversity Archive

Implements the MAP-Elites algorithm for maintaining a diverse population
of high-quality solutions across different behavioral niches.

Reference:
    Mouret & Clune (2015). "Illuminating search spaces by mapping elites."
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class Candidate:
    """
    A candidate solution in the MAP-Elites archive.
    
    Attributes:
        params: Parameters/weights of the solution.
        fitness: Fitness/quality score.
        descriptor: Behavioral descriptor (normalized to [0, 1]).
        generation: Generation when this candidate was created.
        metadata: Additional information (metrics, etc.).
    """
    params: Any
    fitness: float
    descriptor: List[float]
    generation: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Handle numpy arrays
        if isinstance(self.params, np.ndarray):
            data['params'] = self.params.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Candidate':
        """Create from dictionary."""
        # Convert lists back to numpy arrays if needed
        if isinstance(data.get('params'), list):
            data['params'] = np.array(data['params'])
        return cls(**data)


class MAPElites:
    """
    MAP-Elites: Maintains a grid of elite solutions across behavioral space.
    
    The archive is a multi-dimensional grid where each cell represents
    a behavioral niche. Only the best solution per niche is kept.
    
    Args:
        grid_size: Tuple of grid dimensions (e.g., (12, 12) for 2D).
        descriptor_bounds: List of (min, max) tuples for each dimension.
                          If None, assumes [0, 1] for all.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, ...],
        descriptor_bounds: Optional[List[Tuple[float, float]]] = None
    ):
        self.grid_size = grid_size
        self.n_dims = len(grid_size)
        
        if descriptor_bounds is None:
            self.descriptor_bounds = [(0.0, 1.0)] * self.n_dims
        else:
            assert len(descriptor_bounds) == self.n_dims
            self.descriptor_bounds = descriptor_bounds
        
        # Archive: grid of candidates
        self.archive: Dict[Tuple[int, ...], Candidate] = {}
        
        # Best overall solution
        self.best_global: Optional[Candidate] = None
        
        # Statistics
        self.n_insertions = 0
        self.n_replacements = 0
        self.coverage = 0.0
    
    def descriptor_to_index(self, descriptor: List[float]) -> Optional[Tuple[int, ...]]:
        """
        Convert continuous descriptor to discrete grid index.
        
        Args:
            descriptor: Behavioral descriptor (list of floats).
            
        Returns:
            Grid index tuple, or None if out of bounds.
        """
        if len(descriptor) != self.n_dims:
            raise ValueError(
                f"Descriptor has {len(descriptor)} dimensions, "
                f"expected {self.n_dims}"
            )
        
        index = []
        for i, value in enumerate(descriptor):
            min_val, max_val = self.descriptor_bounds[i]
            
            # Clamp to bounds
            value = max(min_val, min(max_val, value))
            
            # Normalize to [0, 1]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.5
            
            # Convert to grid index
            idx = int(normalized * self.grid_size[i])
            idx = min(idx, self.grid_size[i] - 1)  # Clamp to valid range
            index.append(idx)
        
        return tuple(index)
    
    def add(self, candidate: Candidate) -> bool:
        """
        Add a candidate to the archive.
        
        The candidate is added if:
        1. Its niche is empty, OR
        2. Its fitness is better than the current elite in that niche.
        
        Args:
            candidate: Candidate to add.
            
        Returns:
            True if the candidate was added, False otherwise.
        """
        index = self.descriptor_to_index(candidate.descriptor)
        
        if index is None:
            return False
        
        # Update best global
        if self.best_global is None or candidate.fitness > self.best_global.fitness:
            self.best_global = candidate
        
        # Check if we should add this candidate
        if index not in self.archive:
            # Empty niche - add
            self.archive[index] = candidate
            self.n_insertions += 1
            self._update_coverage()
            return True
        else:
            # Niche occupied - compare fitness
            current = self.archive[index]
            if candidate.fitness > current.fitness:
                self.archive[index] = candidate
                self.n_replacements += 1
                return True
        
        return False
    
    def sample(self, n: int, method: str = "uniform") -> List[Candidate]:
        """
        Sample candidates from the archive.
        
        Args:
            n: Number of candidates to sample.
            method: Sampling method ("uniform" or "fitness").
            
        Returns:
            List of sampled candidates.
        """
        if not self.archive:
            return []
        
        elites = list(self.archive.values())
        n = min(n, len(elites))
        
        if method == "uniform":
            indices = np.random.choice(len(elites), size=n, replace=False)
            return [elites[i] for i in indices]
        elif method == "fitness":
            # Sample proportional to fitness
            fitnesses = np.array([e.fitness for e in elites])
            # Shift to positive if needed
            if fitnesses.min() < 0:
                fitnesses = fitnesses - fitnesses.min() + 1e-6
            probs = fitnesses / fitnesses.sum()
            indices = np.random.choice(len(elites), size=n, replace=False, p=probs)
            return [elites[i] for i in indices]
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get archive statistics.
        
        Returns:
            Dictionary with statistics.
        """
        if not self.archive:
            return {
                "coverage": 0.0,
                "n_elites": 0,
                "best_fitness": None,
                "mean_fitness": None,
                "std_fitness": None,
                "n_insertions": self.n_insertions,
                "n_replacements": self.n_replacements,
            }
        
        fitnesses = [c.fitness for c in self.archive.values()]
        
        return {
            "coverage": self.coverage,
            "n_elites": len(self.archive),
            "best_fitness": self.best_global.fitness if self.best_global else None,
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "median_fitness": np.median(fitnesses),
            "n_insertions": self.n_insertions,
            "n_replacements": self.n_replacements,
        }
    
    def _update_coverage(self):
        """Update coverage statistic (percentage of cells filled)."""
        total_cells = np.prod(self.grid_size)
        self.coverage = len(self.archive) / total_cells
    
    def get_best(self) -> Optional[Candidate]:
        """Get the best overall candidate."""
        return self.best_global
    
    def get_all_elites(self) -> List[Candidate]:
        """Get all elite candidates."""
        return list(self.archive.values())
    
    def clear(self):
        """Clear the archive."""
        self.archive.clear()
        self.best_global = None
        self.n_insertions = 0
        self.n_replacements = 0
        self.coverage = 0.0
    
    def to_dict(self) -> Dict:
        """Serialize archive to dictionary."""
        return {
            "grid_size": self.grid_size,
            "descriptor_bounds": self.descriptor_bounds,
            "archive": {
                str(k): v.to_dict() for k, v in self.archive.items()
            },
            "best_global": self.best_global.to_dict() if self.best_global else None,
            "statistics": self.get_statistics(),
        }
    
    def save(self, filepath: str):
        """Save archive to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MAPElites':
        """Load archive from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        archive = cls(
            grid_size=tuple(data['grid_size']),
            descriptor_bounds=data['descriptor_bounds']
        )
        
        # Restore archive
        for key_str, candidate_dict in data['archive'].items():
            key = eval(key_str)  # Convert string back to tuple
            archive.archive[key] = Candidate.from_dict(candidate_dict)
        
        # Restore best global
        if data['best_global']:
            archive.best_global = Candidate.from_dict(data['best_global'])
        
        archive._update_coverage()
        
        return archive
