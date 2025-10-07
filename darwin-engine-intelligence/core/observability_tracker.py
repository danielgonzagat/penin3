"""
Observability Tracker: Comprehensive metrics and logging for evolutionary systems.

Tracks key metrics for Quality-Diversity, Pareto optimization, and general evolution:
- Coverage, QD-score, archive entropy
- Hypervolume, Pareto front size
- Novelty, diversity, stagnation
- Performance statistics
- Component-level metrics

Provides JSON export, time-series data, and real-time monitoring.
"""

import time
import json
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque


@dataclass
class Snapshot:
    """A snapshot of system state at a point in time."""
    timestamp: float
    iteration: int
    
    # QD metrics
    coverage: float = 0.0
    qd_score: float = 0.0
    archive_size: int = 0
    archive_entropy: float = 0.0
    
    # Fitness metrics
    max_fitness: float = 0.0
    mean_fitness: float = 0.0
    min_fitness: float = 0.0
    std_fitness: float = 0.0
    
    # Pareto metrics (if multi-objective)
    hypervolume: float = 0.0
    pareto_front_size: int = 0
    
    # Diversity metrics
    novelty_mean: float = 0.0
    novelty_max: float = 0.0
    behavioral_diversity: float = 0.0
    
    # Performance
    evaluations: int = 0
    evaluations_per_second: float = 0.0
    
    # Stagnation
    generations_without_improvement: int = 0
    best_fitness_plateau: int = 0
    
    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentMetrics:
    """Metrics for a specific component (emitter, island, etc.)."""
    component_id: str
    component_type: str
    
    emissions: int = 0
    improvements: int = 0
    new_discoveries: int = 0
    fitness_gain: float = 0.0
    
    # Time stats
    total_time_seconds: float = 0.0
    avg_time_per_emission: float = 0.0
    
    custom: Dict[str, Any] = field(default_factory=dict)


class ObservabilityTracker:
    """
    Tracks comprehensive metrics for evolutionary systems.
    
    Features:
    - Real-time metric tracking
    - Time-series data storage
    - JSON export
    - Stagnation detection
    - Component-level metrics
    - Custom metric registration
    """
    
    def __init__(self, window_size: int = 100, track_history: bool = True):
        """
        Initialize tracker.
        
        Args:
            window_size: Size of sliding window for moving averages
            track_history: Whether to store full time-series history
        """
        self.window_size = window_size
        self.track_history = track_history
        
        # Current state
        self.current_snapshot: Optional[Snapshot] = None
        
        # History
        self.snapshots: List[Snapshot] = []
        
        # Component metrics
        self.components: Dict[str, ComponentMetrics] = {}
        
        # Sliding windows for moving averages
        self.fitness_window: deque = deque(maxlen=window_size)
        self.coverage_window: deque = deque(maxlen=window_size)
        
        # Stagnation tracking
        self.best_fitness_ever = float('-inf')
        self.generations_since_improvement = 0
        self.best_fitness_history: deque = deque(maxlen=window_size)
        
        # Timing
        self.start_time = time.time()
        self.last_snapshot_time = self.start_time
        
        # Custom metrics registry
        self.custom_metrics: Dict[str, Any] = {}
    
    def record_snapshot(
        self,
        iteration: int,
        archive: Optional[Dict[int, Any]] = None,
        population: Optional[List[Any]] = None,
        pareto_front: Optional[List[Any]] = None,
        **kwargs
    ) -> Snapshot:
        """
        Record a snapshot of current system state.
        
        Args:
            iteration: Current iteration number
            archive: QD archive (dict[niche_id -> individual])
            population: Current population
            pareto_front: Current Pareto front (if multi-objective)
            **kwargs: Additional custom metrics
        
        Returns:
            Snapshot object
        """
        now = time.time()
        
        snapshot = Snapshot(
            timestamp=now,
            iteration=iteration
        )
        
        # QD metrics
        if archive is not None:
            snapshot.archive_size = len(archive)
            
            # Coverage (assuming n_niches is known via custom metric)
            if 'n_niches' in self.custom_metrics:
                snapshot.coverage = len(archive) / max(1, self.custom_metrics['n_niches'])
            
            # QD-score
            fitnesses = [ind.fitness if hasattr(ind, 'fitness') else ind.get('fitness', 0)
                        for ind in archive.values()]
            snapshot.qd_score = sum(fitnesses)
            
            # Fitness stats
            if fitnesses:
                snapshot.max_fitness = max(fitnesses)
                snapshot.mean_fitness = sum(fitnesses) / len(fitnesses)
                snapshot.min_fitness = min(fitnesses)
                snapshot.std_fitness = self._std(fitnesses)
            
            # Archive entropy (behavioral diversity proxy)
            snapshot.archive_entropy = math.log(max(1, len(archive)))
        
        # Population metrics
        if population is not None:
            pop_fitnesses = [ind.fitness if hasattr(ind, 'fitness') else ind.get('fitness', 0)
                            for ind in population]
            if pop_fitnesses:
                snapshot.max_fitness = max(pop_fitnesses)
                snapshot.mean_fitness = sum(pop_fitnesses) / len(pop_fitnesses)
                snapshot.min_fitness = min(pop_fitnesses)
                snapshot.std_fitness = self._std(pop_fitnesses)
        
        # Pareto metrics
        if pareto_front is not None:
            snapshot.pareto_front_size = len(pareto_front)
            # Hypervolume would require reference point
            if 'hypervolume' in kwargs:
                snapshot.hypervolume = kwargs.pop('hypervolume')
        
        # Performance
        time_delta = now - self.last_snapshot_time
        if 'evaluations' in kwargs:
            snapshot.evaluations = kwargs.pop('evaluations')
            if time_delta > 0:
                snapshot.evaluations_per_second = snapshot.evaluations / time_delta
        
        # Stagnation detection
        if snapshot.max_fitness > self.best_fitness_ever:
            self.best_fitness_ever = snapshot.max_fitness
            self.generations_since_improvement = 0
        else:
            self.generations_since_improvement += 1
        
        snapshot.generations_without_improvement = self.generations_since_improvement
        
        # Best fitness plateau detection
        self.best_fitness_history.append(snapshot.max_fitness)
        if len(self.best_fitness_history) >= 10:
            recent_best = list(self.best_fitness_history)[-10:]
            if max(recent_best) - min(recent_best) < 1e-6:
                snapshot.best_fitness_plateau = 10
        
        # Custom metrics
        snapshot.custom = kwargs
        for key, value in self.custom_metrics.items():
            if key not in snapshot.custom:
                snapshot.custom[key] = value
        
        # Update windows
        self.fitness_window.append(snapshot.mean_fitness)
        self.coverage_window.append(snapshot.coverage)
        
        # Store
        self.current_snapshot = snapshot
        if self.track_history:
            self.snapshots.append(snapshot)
        
        self.last_snapshot_time = now
        
        return snapshot
    
    def register_component(
        self,
        component_id: str,
        component_type: str
    ) -> ComponentMetrics:
        """Register a component for tracking."""
        if component_id not in self.components:
            self.components[component_id] = ComponentMetrics(
                component_id=component_id,
                component_type=component_type
            )
        return self.components[component_id]
    
    def update_component(
        self,
        component_id: str,
        emissions: int = 0,
        improvements: int = 0,
        new_discoveries: int = 0,
        fitness_gain: float = 0.0,
        time_seconds: float = 0.0,
        **kwargs
    ):
        """Update component metrics."""
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not registered")
        
        comp = self.components[component_id]
        comp.emissions += emissions
        comp.improvements += improvements
        comp.new_discoveries += new_discoveries
        comp.fitness_gain += fitness_gain
        comp.total_time_seconds += time_seconds
        
        if comp.emissions > 0:
            comp.avg_time_per_emission = comp.total_time_seconds / comp.emissions
        
        for key, value in kwargs.items():
            comp.custom[key] = value
    
    def set_custom_metric(self, key: str, value: Any):
        """Set a custom metric."""
        self.custom_metrics[key] = value
    
    def get_moving_average(self, metric: str, window: Optional[int] = None) -> float:
        """Get moving average of a metric."""
        window = window or self.window_size
        
        if metric == 'fitness':
            data = list(self.fitness_window)
        elif metric == 'coverage':
            data = list(self.coverage_window)
        else:
            # Extract from snapshots
            if not self.snapshots:
                return 0.0
            recent = self.snapshots[-window:]
            data = [getattr(snap, metric, 0.0) for snap in recent]
        
        if not data:
            return 0.0
        return sum(data) / len(data)
    
    def is_stagnating(self, threshold: int = 50) -> bool:
        """Check if evolution is stagnating."""
        return self.generations_since_improvement >= threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        if not self.current_snapshot:
            return {'error': 'No snapshots recorded'}
        
        snap = self.current_snapshot
        
        summary = {
            'current': asdict(snap),
            'moving_averages': {
                'fitness': self.get_moving_average('fitness'),
                'coverage': self.get_moving_average('coverage')
            },
            'stagnation': {
                'is_stagnating': self.is_stagnating(),
                'generations_since_improvement': self.generations_since_improvement,
                'best_fitness_ever': self.best_fitness_ever
            },
            'components': {
                cid: asdict(comp) for cid, comp in self.components.items()
            },
            'runtime_seconds': time.time() - self.start_time
        }
        
        return summary
    
    def export_json(self, filepath: str):
        """Export all data to JSON file."""
        data = {
            'snapshots': [asdict(s) for s in self.snapshots],
            'components': {cid: asdict(comp) for cid, comp in self.components.items()},
            'custom_metrics': self.custom_metrics,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


# ============================================================================
# TEST
# ============================================================================

def test_observability_tracker():
    """Test observability tracker."""
    print("\n" + "="*80)
    print("🧪 TESTE: Observability Tracker")
    print("="*80)
    
    tracker = ObservabilityTracker(window_size=10)
    tracker.set_custom_metric('n_niches', 50)
    
    # Register components
    tracker.register_component('emitter_1', 'improvement')
    tracker.register_component('emitter_2', 'exploration')
    
    # Simulate evolution
    for iteration in range(20):
        # Mock archive
        archive = {
            i: type('Individual', (), {'fitness': 10 + iteration * 0.5 + i * 0.1})()
            for i in range(min(50, 10 + iteration * 2))
        }
        
        # Record snapshot
        snapshot = tracker.record_snapshot(
            iteration=iteration,
            archive=archive,
            evaluations=100 + iteration * 10
        )
        
        # Update components
        tracker.update_component(
            'emitter_1',
            emissions=5,
            improvements=2,
            new_discoveries=1,
            fitness_gain=0.5
        )
        tracker.update_component(
            'emitter_2',
            emissions=5,
            improvements=1,
            new_discoveries=2,
            fitness_gain=0.3
        )
        
        if iteration % 5 == 0:
            print(f"[Obs] iter={iteration:03d} coverage={snapshot.coverage:.3f} "
                  f"qd_score={snapshot.qd_score:.2f} max_fit={snapshot.max_fitness:.4f}")
    
    # Get summary
    summary = tracker.get_summary()
    
    print("\n📊 Sumário Final:")
    print(f"   Iterations: {summary['current']['iteration']}")
    print(f"   Coverage: {summary['current']['coverage']:.3f}")
    print(f"   QD-Score: {summary['current']['qd_score']:.2f}")
    print(f"   Max Fitness: {summary['current']['max_fitness']:.4f}")
    print(f"   Stagnating: {summary['stagnation']['is_stagnating']}")
    print(f"   Runtime: {summary['runtime_seconds']:.2f}s")
    
    print("\n   📈 Component Stats:")
    for cid, comp in summary['components'].items():
        print(f"      {cid}: {comp['emissions']} emissions, "
              f"{comp['improvements']} improvements, {comp['new_discoveries']} discoveries")
    
    # Export
    tracker.export_json('/tmp/observability_test.json')
    print("\n   💾 Exported to: /tmp/observability_test.json")
    
    # Validate
    assert len(tracker.snapshots) == 20, "Missing snapshots"
    assert summary['current']['coverage'] > 0.5, "Coverage baixa"
    
    print("\n✅ Observability Tracker: PASS")
    print("="*80)


if __name__ == "__main__":
    test_observability_tracker()
    print("\n" + "="*80)
    print("✅ observability_tracker.py está FUNCIONAL!")
    print("="*80)
