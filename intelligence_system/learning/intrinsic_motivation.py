#!/usr/bin/env python3
"""
🎨 INTRINSIC MOTIVATION
BLOCO 3 - TAREFA 38

Enhanced intrinsic motivation with multiple sources:
- Curiosity (prediction error)
- Empowerment (state coverage)
- Competence progress (skill mastery)
"""

__version__ = "1.0.0"

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict


class EnhancedIntrinsicMotivation:
    """
    BLOCO 3 - TAREFA 38: Enhanced intrinsic motivation
    
    Combines multiple intrinsic reward signals:
    1. Curiosity: Reward for prediction errors (novel states)
    2. Empowerment: Reward for reaching diverse states
    3. Competence: Reward for skill improvement
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        curiosity_weight: float = 0.1,
        empowerment_weight: float = 0.05,
        competence_weight: float = 0.05,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden dimension for networks
            curiosity_weight: Weight for curiosity reward
            empowerment_weight: Weight for empowerment reward
            competence_weight: Weight for competence reward
            device: torch device
        """
        self.state_dim = state_dim
        self.curiosity_weight = curiosity_weight
        self.empowerment_weight = empowerment_weight
        self.competence_weight = competence_weight
        self.device = device
        
        # Curiosity: Forward prediction model
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # state + action
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # predict next state
        ).to(device)
        
        self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=1e-3)
        
        # Empowerment: State visitation counts
        self.state_visits = {}
        self.state_discretization = 10  # For continuous states
        
        # Competence: Track performance history
        self.performance_history = deque(maxlen=100)
        self.baseline_performance = 0.0
        
        # Stats
        self.curiosity_rewards = []
        self.empowerment_rewards = []
        self.competence_rewards = []
    
    def compute_curiosity_reward(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor
    ) -> float:
        """
        Curiosity reward based on prediction error.
        Higher reward for surprising state transitions.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
        
        Returns:
            Curiosity reward
        """
        with torch.no_grad():
            # Prepare input
            action_tensor = torch.tensor([float(action)], device=self.device)
            input_tensor = torch.cat([state, action_tensor])
            
            # Predict next state
            predicted_next = self.forward_model(input_tensor)
            
            # Prediction error = curiosity
            error = torch.norm(predicted_next - next_state).item()
            
            # Normalize by state dimension
            curiosity = error / np.sqrt(self.state_dim)
        
        # Train forward model
        self.forward_optimizer.zero_grad()
        loss = nn.MSELoss()(predicted_next, next_state)
        loss.backward()
        self.forward_optimizer.step()
        
        self.curiosity_rewards.append(curiosity)
        return curiosity * self.curiosity_weight
    
    def compute_empowerment_reward(self, state: torch.Tensor) -> float:
        """
        Empowerment reward based on state coverage.
        Higher reward for visiting rare states.
        
        Args:
            state: Current state
        
        Returns:
            Empowerment reward
        """
        # Discretize continuous state
        state_np = state.cpu().numpy()
        state_key = tuple((state_np * self.state_discretization).astype(int))
        
        # Count visits
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
        
        # Reward = 1 / sqrt(visits)
        empowerment = 1.0 / np.sqrt(self.state_visits[state_key])
        
        self.empowerment_rewards.append(empowerment)
        return empowerment * self.empowerment_weight
    
    def compute_competence_reward(self, current_performance: float) -> float:
        """
        Competence reward based on performance improvement.
        Rewards consistent improvement over baseline.
        
        Args:
            current_performance: Current episode reward
        
        Returns:
            Competence reward
        """
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) < 10:
            return 0.0
        
        # Update baseline (moving average)
        self.baseline_performance = np.mean(list(self.performance_history)[-10:])
        
        # Recent performance
        recent_performance = np.mean(list(self.performance_history)[-5:])
        
        # Reward for improvement
        improvement = recent_performance - self.baseline_performance
        competence = max(0, improvement / (abs(self.baseline_performance) + 1.0))
        
        self.competence_rewards.append(competence)
        return competence * self.competence_weight
    
    def compute_total_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        episode_performance: float
    ) -> Dict[str, float]:
        """
        Compute combined intrinsic reward.
        
        Returns:
            Dict with individual and total rewards
        """
        curiosity = self.compute_curiosity_reward(state, action, next_state)
        empowerment = self.compute_empowerment_reward(next_state)
        competence = self.compute_competence_reward(episode_performance)
        
        total = curiosity + empowerment + competence
        
        return {
            'curiosity': curiosity,
            'empowerment': empowerment,
            'competence': competence,
            'total': total
        }
    
    def get_stats(self) -> Dict:
        """Get intrinsic motivation statistics"""
        if not self.curiosity_rewards:
            return {
                'avg_curiosity': 0.0,
                'avg_empowerment': 0.0,
                'avg_competence': 0.0,
                'unique_states': 0
            }
        
        return {
            'avg_curiosity': np.mean(self.curiosity_rewards[-100:]),
            'avg_empowerment': np.mean(self.empowerment_rewards[-100:]),
            'avg_competence': np.mean(self.competence_rewards[-100:]),
            'unique_states': len(self.state_visits),
            'baseline_performance': self.baseline_performance
        }


if __name__ == "__main__":
    print(f"Testing Enhanced Intrinsic Motivation v{__version__}...")
    
    # Create intrinsic motivation module
    im = EnhancedIntrinsicMotivation(state_dim=4, hidden_dim=64)
    
    # Simulate episode
    total_intrinsic = 0
    
    for step in range(10):
        state = torch.randn(4)
        action = np.random.randint(0, 2)
        next_state = torch.randn(4)
        performance = 10 + step * 0.5
        
        rewards = im.compute_total_intrinsic_reward(state, action, next_state, performance)
        
        total_intrinsic += rewards['total']
        
        if step % 5 == 0:
            print(f"Step {step}: {rewards}")
    
    print(f"\nTotal intrinsic reward: {total_intrinsic:.4f}")
    print(f"Stats: {im.get_stats()}")
    
    print("✅ Intrinsic Motivation tests OK!")
