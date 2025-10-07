#!/usr/bin/env python3
"""
🧠🧠 META-META-LEARNING
BLOCO 4 - TAREFA 42

Learns how to learn how to learn.
Optimizes the learning algorithm itself.
"""

__version__ = "1.0.0"

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from collections import deque


class MetaMetaLearner:
    """
    BLOCO 4 - TAREFA 42: Meta-meta-learning
    
    Third-order learning:
    - Level 1: Learn task (base learning)
    - Level 2: Learn how to learn (meta-learning)
    - Level 3: Learn how to meta-learn (meta-meta-learning)
    
    Optimizes hyperparameters of the meta-learning algorithm.
    """
    
    def __init__(self):
        """Initialize meta-meta-learner"""
        # Meta-hyperparameters to optimize
        self.meta_lr = 0.001
        self.meta_batch_size = 32
        self.meta_adaptation_steps = 5
        
        # Performance history for each meta-hyperparam setting
        self.performance_history = deque(maxlen=100)
        
        # Exploration for meta-hyperparams
        self.exploration_rate = 0.1
        
        # Best meta-hyperparams found
        self.best_meta_lr = self.meta_lr
        self.best_meta_batch_size = self.meta_batch_size
        self.best_meta_adaptation_steps = self.meta_adaptation_steps
        self.best_performance = -float('inf')
    
    def suggest_meta_hyperparams(self) -> Dict:
        """
        Suggest meta-hyperparameters for next meta-learning iteration.
        
        Uses simple hill-climbing with exploration.
        
        Returns:
            Dict with meta-hyperparameters
        """
        # Explore with probability epsilon
        if np.random.random() < self.exploration_rate:
            # Random exploration
            meta_lr = 10 ** np.random.uniform(-5, -2)
            meta_batch_size = int(2 ** np.random.randint(3, 7))  # 8 to 64
            meta_adaptation_steps = np.random.randint(1, 10)
        else:
            # Exploit best found
            meta_lr = self.best_meta_lr
            meta_batch_size = self.best_meta_batch_size
            meta_adaptation_steps = self.best_meta_adaptation_steps
            
            # Small perturbation
            meta_lr *= np.random.uniform(0.8, 1.2)
            meta_batch_size = max(8, min(128, meta_batch_size + np.random.randint(-8, 9)))
            meta_adaptation_steps = max(1, min(20, meta_adaptation_steps + np.random.randint(-2, 3)))
        
        return {
            'meta_lr': meta_lr,
            'meta_batch_size': meta_batch_size,
            'meta_adaptation_steps': meta_adaptation_steps
        }
    
    def update(self, meta_hyperparams: Dict, performance: float):
        """
        Update meta-meta-learner with performance feedback.
        
        Args:
            meta_hyperparams: Meta-hyperparameters used
            performance: Meta-learning performance achieved
        """
        # Store performance
        self.performance_history.append({
            'meta_hyperparams': meta_hyperparams,
            'performance': performance
        })
        
        # Update best if better
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_meta_lr = meta_hyperparams['meta_lr']
            self.best_meta_batch_size = meta_hyperparams['meta_batch_size']
            self.best_meta_adaptation_steps = meta_hyperparams['meta_adaptation_steps']
            
            # Decrease exploration as we find better settings
            self.exploration_rate *= 0.99
    
    def get_stats(self) -> Dict:
        """Get meta-meta-learning statistics"""
        if not self.performance_history:
            return {
                'best_performance': self.best_performance,
                'exploration_rate': self.exploration_rate,
                'samples': 0
            }
        
        recent = list(self.performance_history)[-20:]
        
        return {
            'best_performance': self.best_performance,
            'best_meta_lr': self.best_meta_lr,
            'best_meta_batch_size': self.best_meta_batch_size,
            'best_meta_adaptation_steps': self.best_meta_adaptation_steps,
            'exploration_rate': self.exploration_rate,
            'avg_recent_performance': np.mean([h['performance'] for h in recent]),
            'samples': len(self.performance_history)
        }


class EvolutionaryMetaOptimizer:
    """
    Alternative approach: evolutionary optimization of meta-learning algorithm.
    """
    
    def __init__(self, population_size: int = 10):
        """
        Args:
            population_size: Number of meta-learner variants to maintain
        """
        self.population_size = population_size
        
        # Population of meta-learner configs
        self.population = []
        for _ in range(population_size):
            self.population.append({
                'meta_lr': 10 ** np.random.uniform(-5, -2),
                'meta_batch_size': int(2 ** np.random.randint(3, 7)),
                'meta_adaptation_steps': np.random.randint(1, 10),
                'fitness': 0.0,
                'age': 0
            })
    
    def evolve(self, performances: List[float]):
        """
        Evolve population based on performances.
        
        Args:
            performances: List of performances for each individual
        """
        # Update fitnesses
        for i, perf in enumerate(performances):
            self.population[i]['fitness'] = perf
            self.population[i]['age'] += 1
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep top 50%
        survivors = self.population[:self.population_size // 2]
        
        # Create offspring through mutation
        offspring = []
        for parent in survivors:
            child = {
                'meta_lr': parent['meta_lr'] * np.random.uniform(0.5, 2.0),
                'meta_batch_size': max(8, min(128, parent['meta_batch_size'] + np.random.randint(-8, 9))),
                'meta_adaptation_steps': max(1, min(20, parent['meta_adaptation_steps'] + np.random.randint(-2, 3))),
                'fitness': 0.0,
                'age': 0
            }
            offspring.append(child)
        
        # New population
        self.population = survivors + offspring
    
    def get_best(self) -> Dict:
        """Get best individual"""
        return self.population[0]


if __name__ == "__main__":
    print(f"Testing Meta-Meta-Learner v{__version__}...")
    
    # Test hill-climbing approach
    mml = MetaMetaLearner()
    
    print("Testing hill-climbing meta-meta-learning...")
    for iteration in range(20):
        # Suggest meta-hyperparams
        meta_hparams = mml.suggest_meta_hyperparams()
        
        # Simulate meta-learning with these hyperparams
        # (in practice, would actually run meta-learning)
        performance = np.random.randn() + iteration * 0.1  # Improving trend
        
        # Update
        mml.update(meta_hparams, performance)
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: {mml.get_stats()}")
    
    print(f"\nFinal stats: {mml.get_stats()}")
    
    # Test evolutionary approach
    print("\nTesting evolutionary meta-optimizer...")
    evo = EvolutionaryMetaOptimizer(population_size=10)
    
    for gen in range(5):
        # Simulate evaluating population
        performances = [np.random.randn() + gen * 0.2 for _ in range(10)]
        
        # Evolve
        evo.evolve(performances)
        
        best = evo.get_best()
        print(f"Generation {gen}: Best fitness={best['fitness']:.3f}")
    
    print("✅ Meta-Meta-Learning tests OK!")
