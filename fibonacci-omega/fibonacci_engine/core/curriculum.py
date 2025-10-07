"""
Fibonacci Curriculum: Progressive Task Sampling

Implements a curriculum learning system where the number of tasks/episodes
grows according to the Fibonacci sequence, and difficulty can be
progressively increased.
"""

import numpy as np
from typing import List, Any, Callable, Optional, Dict
from fibonacci_engine.core.math_utils import fibonacci_seq, fibonacci_window


class FibonacciCurriculum:
    """
    Curriculum learning with Fibonacci-based task sampling.
    
    The number of tasks/episodes sampled at each generation follows
    the Fibonacci sequence, promoting gradual increase in evaluation budget.
    
    Optionally supports difficulty progression (easy -> medium -> hard).
    
    Args:
        task_sampler: Function that generates tasks given a count and difficulty.
                     Signature: (n: int, difficulty: float) -> List[Any]
        fib_depth: Maximum depth for Fibonacci sequence.
        enable_difficulty: Whether to enable progressive difficulty.
        difficulty_schedule: List of (generation, difficulty) tuples.
                           difficulty ∈ [0, 1], 0 = easiest.
    """
    
    def __init__(
        self,
        task_sampler: Callable[[int, float], List[Any]],
        fib_depth: int = 12,
        enable_difficulty: bool = True,
        difficulty_schedule: Optional[List[tuple]] = None,
    ):
        self.task_sampler = task_sampler
        self.fib_depth = fib_depth
        self.enable_difficulty = enable_difficulty
        
        # Default difficulty schedule: gradual increase
        if difficulty_schedule is None:
            self.difficulty_schedule = [
                (1, 0.0),      # Gen 1-10: easy
                (10, 0.25),    # Gen 10-20: easy-medium
                (20, 0.5),     # Gen 20-40: medium
                (40, 0.75),    # Gen 40+: medium-hard
                (60, 1.0),     # Gen 60+: hard
            ]
        else:
            self.difficulty_schedule = sorted(difficulty_schedule)
        
        self.current_generation = 0
        self.current_difficulty = 0.0
        self.task_history: List[Dict[str, Any]] = []
    
    def get_difficulty(self, generation: int) -> float:
        """
        Get difficulty level for a given generation.
        
        Args:
            generation: Current generation number.
            
        Returns:
            Difficulty level in [0, 1].
        """
        if not self.enable_difficulty:
            return 0.5  # Medium difficulty
        
        # Find applicable difficulty from schedule
        difficulty = self.difficulty_schedule[0][1]
        for gen_threshold, diff in self.difficulty_schedule:
            if generation >= gen_threshold:
                difficulty = diff
            else:
                break
        
        return difficulty
    
    def sample_tasks(self, generation: int) -> List[Any]:
        """
        Sample tasks for a given generation.
        
        The number of tasks follows the Fibonacci sequence.
        
        Args:
            generation: Current generation number.
            
        Returns:
            List of tasks.
        """
        self.current_generation = generation
        
        # Get Fibonacci window size
        n_tasks = fibonacci_window(generation, self.fib_depth)
        
        # Get current difficulty
        self.current_difficulty = self.get_difficulty(generation)
        
        # Sample tasks
        tasks = self.task_sampler(n_tasks, self.current_difficulty)
        
        # Record in history
        self.task_history.append({
            "generation": generation,
            "n_tasks": n_tasks,
            "difficulty": self.current_difficulty,
        })
        
        return tasks
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get curriculum statistics.
        
        Returns:
            Dictionary with statistics.
        """
        if not self.task_history:
            return {
                "current_generation": 0,
                "current_difficulty": 0.0,
                "total_tasks_sampled": 0,
                "mean_tasks_per_gen": 0.0,
            }
        
        total_tasks = sum(h["n_tasks"] for h in self.task_history)
        
        return {
            "current_generation": self.current_generation,
            "current_difficulty": self.current_difficulty,
            "total_generations": len(self.task_history),
            "total_tasks_sampled": total_tasks,
            "mean_tasks_per_gen": total_tasks / len(self.task_history),
            "difficulty_schedule": self.difficulty_schedule,
        }
    
    def reset(self):
        """Reset curriculum state."""
        self.current_generation = 0
        self.current_difficulty = 0.0
        self.task_history.clear()


def default_task_sampler(n: int, difficulty: float) -> List[Dict[str, Any]]:
    """
    Default task sampler for testing/examples.
    
    Generates synthetic tasks with varying complexity based on difficulty.
    
    Args:
        n: Number of tasks to generate.
        difficulty: Difficulty level in [0, 1].
        
    Returns:
        List of task dictionaries.
    """
    tasks = []
    
    for i in range(n):
        # Task complexity increases with difficulty
        dim = int(10 + difficulty * 90)  # 10 to 100 dimensions
        noise = 0.5 - difficulty * 0.4   # 0.5 to 0.1 noise
        
        task = {
            "id": i,
            "type": "synthetic",
            "dim": dim,
            "noise": noise,
            "target": np.random.randn(dim).tolist(),
            "difficulty": difficulty,
        }
        tasks.append(task)
    
    return tasks
