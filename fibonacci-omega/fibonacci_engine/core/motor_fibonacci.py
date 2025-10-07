"""
Motor Fibonacci: Universal AI Optimization Engine

The main engine that orchestrates all components:
- Fibonacci scheduling
- MAP-Elites quality-diversity
- Multi-scale spiral search
- Meta-controller (UCB bandit)
- Curriculum learning
- WORM ledger for auditability
- Automatic rollback on regression
"""

import json
import time
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from fibonacci_engine.core.map_elites import MAPElites, Candidate
from fibonacci_engine.core.meta_controller import MetaController
from fibonacci_engine.core.curriculum import FibonacciCurriculum
from fibonacci_engine.core.worm_ledger import WormLedger
from fibonacci_engine.core.rollback_guard import RollbackGuard
from fibonacci_engine.core.math_utils import (
    fibonacci_seq,
    fibonacci_window,
    phi_mix,
    phi_mix_array,
    spiral_scales,
    explore_exploit_budget,
    golden_ratio,
)


@dataclass
class FibonacciConfig:
    """
    Configuration for Fibonacci Engine.
    
    Attributes:
        max_generations: Maximum number of generations to run.
        fib_depth: Depth of Fibonacci sequence (max index).
        population: Population size per generation.
        elites_grid: Grid size for MAP-Elites (e.g., (12, 12)).
        rollback_delta: Threshold for rollback trigger (fitness drop).
        seed: Random seed for reproducibility.
        score_key: Key in metrics dict for fitness score.
        meta_control_arms: List of strategy names for meta-controller.
        enable_curriculum: Whether to enable curriculum learning.
        enable_rollback: Whether to enable automatic rollback.
        save_snapshots_every: Save snapshot every N generations (0 = disabled).
        verbose: Verbosity level (0 = quiet, 1 = normal, 2 = detailed).
    """
    max_generations: int = 200
    fib_depth: int = 12
    population: int = 48
    elites_grid: Tuple[int, ...] = (12, 12)
    rollback_delta: float = 0.02
    seed: int = 42
    score_key: str = "fitness"
    meta_control_arms: List[str] = field(
        default_factory=lambda: ["small", "medium", "large", "adaptive"]
    )
    enable_curriculum: bool = True
    enable_rollback: bool = True
    save_snapshots_every: int = 0
    verbose: int = 1
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FibonacciConfig':
        # Convert elites_grid back to tuple
        if 'elites_grid' in data and isinstance(data['elites_grid'], list):
            data['elites_grid'] = tuple(data['elites_grid'])
        return cls(**data)


class FibonacciEngine:
    """
    The main Fibonacci Engine for universal AI optimization.
    
    This engine connects to any host system via adapters and improves
    its policies/models through quality-diversity evolution guided by
    Fibonacci principles and golden ratio.
    
    Args:
        config: Engine configuration.
        evaluate_fn: Function to evaluate candidates.
                    Signature: (params, tasks) -> Dict[str, float]
                    Must return dict with at least config.score_key.
        descriptor_fn: Function to compute behavioral descriptor.
                      Signature: (params, metrics) -> List[float]
                      Returns list of values in [0, 1].
        mutate_fn: Function to mutate parameters.
                  Signature: (params, magnitude) -> params
        cross_fn: Function to crossover two parameter sets.
                 Signature: (params_a, params_b) -> params
        task_sampler: Function to sample tasks.
                     Signature: (n, difficulty) -> List[Any]
        initial_params: Initial parameters to seed the population.
    """
    
    def __init__(
        self,
        config: FibonacciConfig,
        evaluate_fn: Callable[[Any, List[Any]], Dict[str, float]],
        descriptor_fn: Callable[[Any, Dict[str, float]], List[float]],
        mutate_fn: Callable[[Any, float], Any],
        cross_fn: Callable[[Any, Any], Any],
        task_sampler: Callable[[int, float], List[Any]],
        initial_params: Optional[Any] = None,
    ):
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.descriptor_fn = descriptor_fn
        self.mutate_fn = mutate_fn
        self.cross_fn = cross_fn
        self.task_sampler = task_sampler
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Initialize components
        self.archive = MAPElites(grid_size=config.elites_grid)
        self.ledger = WormLedger()
        self.rollback_guard = RollbackGuard(
            rollback_delta=config.rollback_delta,
            window_size=5,
            min_generations=10,
        )
        self.meta_controller = MetaController(
            arm_names=config.meta_control_arms,
            c=1.414,
            initial_pulls=3,
        )
        self.curriculum = FibonacciCurriculum(
            task_sampler=task_sampler,
            fib_depth=config.fib_depth,
            enable_difficulty=config.enable_curriculum,
        )
        
        # State
        self.current_generation = 0
        self.is_running = False
        self.initial_params = initial_params
        
        # Statistics
        self.generation_times: List[float] = []
        self.best_fitness_history: List[float] = []
        
        # Initialize with seed population
        self._initialize_population()
        
        # Log initialization
        self.ledger.append("engine_initialized", {
            "config": config.to_dict(),
            "seed": config.seed,
        })
    
    def _initialize_population(self):
        """Initialize population with random or provided parameters."""
        if self.initial_params is None:
            # Generate random initial parameters
            # Default: 10D random vector
            self.initial_params = np.random.randn(10)
        
        # Evaluate and add initial candidate
        tasks = self.task_sampler(1, 0.0)  # Easy initial task
        metrics = self.evaluate_fn(self.initial_params, tasks)
        fitness = metrics.get(self.config.score_key, 0.0)
        descriptor = self.descriptor_fn(self.initial_params, metrics)
        
        initial_candidate = Candidate(
            params=self.initial_params,
            fitness=fitness,
            descriptor=descriptor,
            generation=0,
            metadata=metrics,
        )
        
        self.archive.add(initial_candidate)
        
        if self.config.verbose >= 1:
            print(f"[Init] Initial fitness: {fitness:.4f}")
        
        self.ledger.append("initial_candidate_added", {
            "fitness": fitness,
            "descriptor": descriptor,
        })
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one generation of the Fibonacci Engine.
        
        Returns:
            Dictionary with generation statistics.
        """
        start_time = time.time()
        self.current_generation += 1
        gen = self.current_generation
        
        if self.config.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Generation {gen}")
            print(f"{'='*60}")
        
        # 1. Sample tasks using curriculum
        tasks = self.curriculum.sample_tasks(gen)
        n_tasks = len(tasks)
        
        # 2. Get Fibonacci window
        window = fibonacci_window(gen, self.config.fib_depth)
        
        # 3. Get explore/exploit budget
        explore_budget, exploit_budget = explore_exploit_budget(
            gen, self.config.population, self.config.fib_depth
        )
        
        # 4. Get spiral scales
        scales = spiral_scales(gen, self.config.fib_depth)
        
        if self.config.verbose >= 2:
            print(f"  Tasks: {n_tasks}")
            print(f"  Window: {window}")
            print(f"  Budget: explore={explore_budget}, exploit={exploit_budget}")
            print(f"  Scales: {scales}")
        
        # 5. Generate offspring
        offspring = []
        
        # Exploration: mutations with different scales
        for i in range(explore_budget):
            # Select scale using meta-controller
            scale_name = self.meta_controller.select_arm()
            
            if scale_name == "small":
                magnitude = scales[0]
            elif scale_name == "medium":
                magnitude = scales[1]
            elif scale_name == "large":
                magnitude = scales[2]
            else:  # adaptive
                # Mix scales using phi
                magnitude = phi_mix(scales[0], scales[2], 0.5)
            
            # Sample parent from archive
            if self.archive.archive:
                parents = self.archive.sample(1, method="uniform")
                parent = parents[0]
                
                # Mutate
                child_params = self.mutate_fn(parent.params, magnitude)
                offspring.append((child_params, scale_name, "mutation"))
        
        # Exploitation: crossover of good solutions
        for i in range(exploit_budget):
            if len(self.archive.archive) >= 2:
                parents = self.archive.sample(2, method="fitness")
                child_params = self.cross_fn(parents[0].params, parents[1].params)
                offspring.append((child_params, "crossover", "crossover"))
        
        # 6. Evaluate offspring
        n_improvements = 0
        scale_rewards = {}
        
        for child_params, strategy, method in offspring:
            # Evaluate
            metrics = self.evaluate_fn(child_params, tasks)
            fitness = metrics.get(self.config.score_key, 0.0)
            descriptor = self.descriptor_fn(child_params, metrics)
            
            candidate = Candidate(
                params=child_params,
                fitness=fitness,
                descriptor=descriptor,
                generation=gen,
                metadata={"method": method, "strategy": strategy},
            )
            
            # Try to add to archive
            was_added = self.archive.add(candidate)
            
            if was_added:
                n_improvements += 1
                
                # Reward the strategy
                if strategy not in scale_rewards:
                    scale_rewards[strategy] = []
                scale_rewards[strategy].append(fitness)
                
                # Log elite addition
                self.ledger.append("elite_added", {
                    "generation": gen,
                    "fitness": fitness,
                    "descriptor": descriptor,
                    "method": method,
                    "strategy": strategy,
                })
        
        # 7. Update meta-controller
        for strategy, rewards in scale_rewards.items():
            # Only update if this is a meta-control arm
            if strategy in self.config.meta_control_arms:
                avg_reward = np.mean(rewards)
                self.meta_controller.update(strategy, avg_reward)
        
        # For meta-control arms with no success, give small penalty
        for strategy in self.config.meta_control_arms:
            if strategy not in scale_rewards:
                self.meta_controller.update(strategy, 0.0)
        
        # 8. Check for rollback
        rollback_info = None
        if self.config.enable_rollback:
            self.rollback_guard.record(gen, self.archive)
            rollback_check = self.rollback_guard.should_rollback(gen, self.archive)
            
            if rollback_check["should_rollback"]:
                rollback_info = rollback_check
                self.ledger.append("rollback_triggered", rollback_info)
                
                if self.config.verbose >= 1:
                    print(f"  ⚠️  ROLLBACK TRIGGERED!")
                    print(f"      Baseline mean: {rollback_check['baseline_mean']:.4f}")
                    print(f"      Current mean: {rollback_check['current_mean']:.4f}")
        
        # 9. Update statistics
        stats = self.archive.get_statistics()
        best_fitness = stats.get("best_fitness", 0.0)
        self.best_fitness_history.append(best_fitness)
        
        elapsed = time.time() - start_time
        self.generation_times.append(elapsed)
        
        # Log generation
        gen_stats = {
            "generation": gen,
            "best_fitness": best_fitness,
            "mean_fitness": stats.get("mean_fitness", 0.0),
            "n_elites": stats.get("n_elites", 0),
            "coverage": stats.get("coverage", 0.0),
            "n_improvements": n_improvements,
            "time": elapsed,
        }
        self.ledger.append("generation_complete", gen_stats)
        
        if self.config.verbose >= 1:
            print(f"  Best: {best_fitness:.4f}")
            print(f"  Mean: {stats.get('mean_fitness', 0.0):.4f}")
            print(f"  Elites: {stats.get('n_elites', 0)}")
            print(f"  Coverage: {stats.get('coverage', 0.0):.2%}")
            print(f"  Improvements: {n_improvements}/{len(offspring)}")
            print(f"  Time: {elapsed:.2f}s")
        
        # Save snapshot if configured
        if self.config.save_snapshots_every > 0:
            if gen % self.config.save_snapshots_every == 0:
                snapshot_path = f"fibonacci_engine/persistence/snapshot_gen{gen}.json"
                self.snapshot(snapshot_path)
        
        return gen_stats
    
    def run(self, generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the engine for multiple generations.
        
        Args:
            generations: Number of generations to run. If None, uses config.
            
        Returns:
            Dictionary with final statistics.
        """
        if generations is None:
            generations = self.config.max_generations
        
        self.is_running = True
        
        try:
            for i in range(generations):
                if not self.is_running:
                    break
                self.step()
        finally:
            self.is_running = False
        
        # Final statistics
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status and statistics.
        
        Returns:
            Dictionary with comprehensive status.
        """
        archive_stats = self.archive.get_statistics()
        curriculum_stats = self.curriculum.get_statistics()
        meta_stats = self.meta_controller.get_all_statistics()
        rollback_stats = self.rollback_guard.get_statistics()
        ledger_stats = self.ledger.get_statistics()
        
        return {
            "generation": self.current_generation,
            "is_running": self.is_running,
            "archive": archive_stats,
            "curriculum": curriculum_stats,
            "meta_controller": meta_stats,
            "rollback_guard": rollback_stats,
            "ledger": ledger_stats,
            "mean_gen_time": np.mean(self.generation_times) if self.generation_times else 0.0,
        }
    
    def snapshot(self, filepath: str):
        """
        Save complete engine snapshot.
        
        Args:
            filepath: Path to save snapshot.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        snapshot = {
            "config": self.config.to_dict(),
            "generation": self.current_generation,
            "archive": self.archive.to_dict(),
            "curriculum": self.curriculum.get_statistics(),
            "meta_controller": self.meta_controller.get_all_statistics(),
            "rollback_guard": self.rollback_guard.get_statistics(),
            "best_fitness_history": self.best_fitness_history,
            "generation_times": self.generation_times,
        }
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        if self.config.verbose >= 1:
            print(f"  💾 Snapshot saved to {filepath}")
    
    def stop(self):
        """Stop the engine (graceful shutdown)."""
        self.is_running = False
        self.ledger.append("engine_stopped", {
            "generation": self.current_generation,
        })
