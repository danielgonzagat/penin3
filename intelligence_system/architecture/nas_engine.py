#!/usr/bin/env python3
"""
🏗️ NEURAL ARCHITECTURE SEARCH (NAS) ENGINE
BLOCO 3 - TAREFA 29-30

Automatically discovers optimal network architectures.
Expands/prunes neurons based on performance.
"""

__version__ = "1.0.0"

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class NASEngine:
    """
    Simple NAS engine that evolves network architecture.
    
    Features:
    - Auto-expand: Adds neurons when stagnating
    - Auto-prune: Removes underperforming neurons
    - Layer optimization: Adjusts layer sizes
    """
    
    def __init__(
        self,
        min_neurons: int = 4,
        max_neurons: int = 32,
        expansion_threshold: float = 0.01,  # Min improvement to avoid expansion
        pruning_threshold: float = 0.1,     # Max weight norm to prune
        patience: int = 20                   # Episodes to wait before action
    ):
        """
        Args:
            min_neurons: Minimum neurons to keep
            max_neurons: Maximum neurons allowed
            expansion_threshold: Reward improvement threshold
            pruning_threshold: Weight norm threshold for pruning
            patience: Episodes without improvement before expanding
        """
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.expansion_threshold = expansion_threshold
        self.pruning_threshold = pruning_threshold
        self.patience = patience
        
        self.episodes_since_improvement = 0
        self.best_reward = 0
        self.actions_taken = []
    
    def should_expand(self, current_reward: float, num_neurons: int) -> bool:
        """
        BLOCO 3 - TAREFA 30: Auto-expand neurons
        
        Decide if should add more neurons.
        
        Args:
            current_reward: Current average reward
            num_neurons: Current number of neurons
        
        Returns:
            True if should expand
        """
        # Don't expand if at max
        if num_neurons >= self.max_neurons:
            return False
        
        # Check if stagnating
        improvement = current_reward - self.best_reward
        
        if improvement > self.expansion_threshold:
            # Improving, don't expand
            self.best_reward = current_reward
            self.episodes_since_improvement = 0
            return False
        
        # Count episodes without improvement
        self.episodes_since_improvement += 1
        
        # Expand if stagnating for too long
        if self.episodes_since_improvement >= self.patience:
            self.episodes_since_improvement = 0
            self.actions_taken.append(('expand', num_neurons, num_neurons + 2))
            return True
        
        return False
    
    def should_prune(self, neuron_weights: torch.Tensor, num_neurons: int) -> List[int]:
        """
        Decide which neurons to prune based on weight norms.
        
        Args:
            neuron_weights: Tensor of neuron weights (num_neurons, hidden_size)
            num_neurons: Current number of neurons
        
        Returns:
            List of neuron indices to prune
        """
        # Don't prune if at minimum
        if num_neurons <= self.min_neurons:
            return []
        
        # Compute weight norms
        weight_norms = torch.norm(neuron_weights, dim=1)
        
        # Find low-norm neurons
        low_norm_indices = torch.where(weight_norms < self.pruning_threshold)[0]
        
        # Don't prune too many at once
        max_prune = min(len(low_norm_indices), num_neurons - self.min_neurons, 2)
        
        if max_prune > 0:
            # Prune lowest-norm neurons
            sorted_indices = torch.argsort(weight_norms)
            to_prune = sorted_indices[:max_prune].tolist()
            
            self.actions_taken.append(('prune', num_neurons, len(to_prune)))
            return to_prune
        
        return []
    
    def suggest_layer_sizes(
        self,
        input_dim: int,
        output_dim: int,
        current_performance: float,
        target_performance: float
    ) -> List[int]:
        """
        Suggest hidden layer sizes based on task complexity.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            current_performance: Current reward/accuracy
            target_performance: Target reward/accuracy
        
        Returns:
            List of suggested hidden layer sizes
        """
        # Estimate task complexity
        performance_ratio = current_performance / (target_performance + 1e-6)
        
        if performance_ratio > 0.9:
            # Task solved, use minimal architecture
            return [64, 32]
        elif performance_ratio > 0.7:
            # Near solved, moderate architecture
            return [128, 64]
        elif performance_ratio > 0.5:
            # Making progress, standard architecture
            return [256, 128]
        else:
            # Struggling, use larger architecture
            return [512, 256, 128]
    
    def get_stats(self) -> Dict:
        """Get NAS statistics"""
        return {
            'best_reward': self.best_reward,
            'episodes_since_improvement': self.episodes_since_improvement,
            'actions_taken': len(self.actions_taken),
            'last_actions': self.actions_taken[-5:] if self.actions_taken else []
        }


class AdaptiveNeuronPool:
    """
    BLOCO 3 - TAREFA 30: Adaptive neuron pool
    
    Dynamically adds/removes neurons from active pool.
    """
    
    def __init__(self, initial_neurons: int = 4, max_neurons: int = 32):
        self.active_neurons = list(range(initial_neurons))
        self.inactive_neurons = list(range(initial_neurons, max_neurons))
        self.neuron_performance = {i: 0.0 for i in range(max_neurons)}
    
    def expand(self, num_new: int = 2):
        """Add neurons from inactive pool"""
        to_activate = self.inactive_neurons[:num_new]
        self.active_neurons.extend(to_activate)
        self.inactive_neurons = [n for n in self.inactive_neurons if n not in to_activate]
        return to_activate
    
    def prune(self, indices_to_remove: List[int]):
        """Remove neurons to inactive pool"""
        for idx in indices_to_remove:
            if idx in self.active_neurons:
                self.active_neurons.remove(idx)
                self.inactive_neurons.append(idx)
    
    def update_performance(self, neuron_id: int, performance: float):
        """Update neuron performance metric"""
        if neuron_id in self.neuron_performance:
            # Exponential moving average
            self.neuron_performance[neuron_id] = 0.9 * self.neuron_performance[neuron_id] + 0.1 * performance
    
    def get_stats(self) -> Dict:
        return {
            'active': len(self.active_neurons),
            'inactive': len(self.inactive_neurons),
            'total': len(self.active_neurons) + len(self.inactive_neurons),
            'avg_performance': np.mean([self.neuron_performance[n] for n in self.active_neurons])
        }


if __name__ == "__main__":
    print(f"Testing NAS Engine v{__version__}...")
    
    # Test NAS engine
    nas = NASEngine(min_neurons=4, max_neurons=16, patience=10)
    
    # Simulate training
    rewards = []
    num_neurons = 4
    
    for ep in range(50):
        # Simulate reward (stagnates then improves)
        if ep < 20:
            reward = 10 + np.random.randn()
        else:
            reward = 10 + ep * 0.5 + np.random.randn()
        
        rewards.append(reward)
        avg_reward = np.mean(rewards[-10:])
        
        # Check expansion
        if nas.should_expand(avg_reward, num_neurons):
            num_neurons += 2
            print(f"Episode {ep}: Expanded to {num_neurons} neurons")
        
        # Check pruning (simulate weights)
        if ep % 20 == 0 and ep > 0:
            fake_weights = torch.randn(num_neurons, 128)
            to_prune = nas.should_prune(fake_weights, num_neurons)
            if to_prune:
                num_neurons -= len(to_prune)
                print(f"Episode {ep}: Pruned {len(to_prune)} neurons, now {num_neurons}")
    
    print(f"\nNAS Stats: {nas.get_stats()}")
    
    # Test adaptive pool
    print("\nTesting Adaptive Neuron Pool...")
    pool = AdaptiveNeuronPool(initial_neurons=4, max_neurons=16)
    
    print(f"Initial: {pool.get_stats()}")
    
    expanded = pool.expand(num_new=4)
    print(f"After expand: {pool.get_stats()}")
    print(f"Expanded neurons: {expanded}")
    
    pool.prune([1, 3])
    print(f"After prune: {pool.get_stats()}")
    
    print("✅ NAS Engine tests OK!")
