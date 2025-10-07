"""
RL Synthetic Adapter: Reinforcement Learning Toy Problem

A simple synthetic RL environment for testing the Fibonacci Engine.
The agent learns a linear policy to navigate towards a target.
"""

import numpy as np
from typing import Dict, List, Any


class RLSyntheticAdapter:
    """
    Adapter for synthetic RL problems.
    
    Environment: Simple 2D navigation task.
    - State: (x, y) position
    - Action: (dx, dy) velocity
    - Goal: Reach target position
    - Reward: Negative distance to target
    
    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        episode_length: Length of each episode.
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        episode_length: int = 50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
    
    def evaluate(self, params: np.ndarray, tasks: List[Any]) -> Dict[str, float]:
        """
        Evaluate policy parameters on RL tasks.
        
        Args:
            params: Policy parameters (linear policy weights).
            tasks: List of task specifications.
            
        Returns:
            Dictionary with 'fitness' and other metrics.
        """
        # Reshape params to policy matrix
        expected_size = self.state_dim * self.action_dim
        if isinstance(params, np.ndarray):
            if params.size != expected_size:
                params = np.random.randn(expected_size)
        else:
            params = np.random.randn(expected_size)
        
        policy = params.reshape(self.action_dim, self.state_dim)
        
        total_reward = 0.0
        episode_rewards = []
        
        for task in tasks:
            # Extract target from task
            if isinstance(task, dict) and "target" in task:
                target = np.array(task["target"][:self.state_dim])
            else:
                target = np.random.randn(self.state_dim)
            
            # Run episode
            state = np.random.randn(self.state_dim) * 0.1  # Start near origin
            episode_reward = 0.0
            
            for step in range(self.episode_length):
                # Linear policy: action = policy @ state
                action = policy @ state
                
                # Clip action to valid range
                action = np.tanh(action)
                
                # Simple dynamics: state update (map action_dim to state_dim)
                dt = 0.1
                # Use sum of actions as a scalar influence
                action_influence = np.mean(action)
                state = state * (1 - dt) + dt * (target + action_influence * 0.1)
                
                # Reward: negative distance to target
                distance = np.linalg.norm(state - target)
                reward = -distance
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
        
        mean_reward = total_reward / len(tasks) if tasks else 0.0
        
        # Normalize fitness to roughly [0, 1]
        fitness = 1.0 / (1.0 + abs(mean_reward) / 10.0)
        
        return {
            "fitness": fitness,
            "mean_reward": mean_reward,
            "std_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "min_reward": np.min(episode_rewards) if episode_rewards else 0.0,
            "max_reward": np.max(episode_rewards) if episode_rewards else 0.0,
        }
    
    def descriptor(self, params: np.ndarray, metrics: Dict[str, float]) -> List[float]:
        """
        Compute behavioral descriptor.
        
        Descriptor dimensions:
        1. Novelty: variance/diversity of behavior
        2. Robustness: consistency across tasks
        
        Args:
            params: Policy parameters.
            metrics: Evaluation metrics.
            
        Returns:
            List of descriptor values in [0, 1].
        """
        # Novelty: based on policy entropy/diversity
        if isinstance(params, np.ndarray):
            param_std = np.std(params)
            novelty = min(1.0, param_std / 2.0)
        else:
            novelty = 0.5
        
        # Robustness: based on consistency (low variance)
        std_reward = metrics.get("std_reward", 1.0)
        robustness = 1.0 / (1.0 + std_reward)
        
        return [novelty, robustness]
    
    def mutate(self, params: np.ndarray, magnitude: float) -> np.ndarray:
        """
        Mutate policy parameters.
        
        Args:
            params: Original parameters.
            magnitude: Mutation magnitude.
            
        Returns:
            Mutated parameters.
        """
        if not isinstance(params, np.ndarray):
            params = np.random.randn(self.state_dim * self.action_dim)
        
        noise = np.random.randn(*params.shape) * magnitude
        return params + noise
    
    def crossover(self, params_a: np.ndarray, params_b: np.ndarray) -> np.ndarray:
        """
        Crossover two parameter sets.
        
        Args:
            params_a: First parent.
            params_b: Second parent.
            
        Returns:
            Child parameters.
        """
        if not isinstance(params_a, np.ndarray):
            params_a = np.random.randn(self.state_dim * self.action_dim)
        if not isinstance(params_b, np.ndarray):
            params_b = np.random.randn(self.state_dim * self.action_dim)
        
        # Ensure same shape
        if params_a.shape != params_b.shape:
            min_size = min(params_a.size, params_b.size)
            params_a = params_a.flatten()[:min_size]
            params_b = params_b.flatten()[:min_size]
        
        # Uniform crossover
        mask = np.random.rand(*params_a.shape) > 0.5
        child = np.where(mask, params_a, params_b)
        
        return child
    
    def task_sampler(self, n: int, difficulty: float) -> List[Dict[str, Any]]:
        """
        Sample RL tasks.
        
        Args:
            n: Number of tasks.
            difficulty: Difficulty level in [0, 1].
            
        Returns:
            List of task specifications.
        """
        tasks = []
        for i in range(n):
            # Target distance increases with difficulty
            target_scale = 1.0 + difficulty * 5.0
            target = np.random.randn(self.state_dim) * target_scale
            
            tasks.append({
                "id": i,
                "type": "navigation",
                "target": target.tolist(),
                "difficulty": difficulty,
            })
        
        return tasks
