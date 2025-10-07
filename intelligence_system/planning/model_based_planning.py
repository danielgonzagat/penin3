#!/usr/bin/env python3
"""
🎯 MODEL-BASED PLANNING
BLOCO 4 - TAREFA 58

Uses world model for lookahead planning.
Simulates future trajectories mentally before acting.
"""

__version__ = "1.0.0"

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '/root')
from intelligence_system.models.world_model import WorldModel


class ModelBasedPlanner:
    """
    BLOCO 4 - TAREFA 58: Model-based planning
    
    Plans actions using world model simulations.
    Methods:
    - Random shooting: sample random action sequences
    - CEM (Cross-Entropy Method): iterative refinement
    - MPC (Model Predictive Control): receding horizon
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 100,
        top_k: int = 10,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            world_model: Trained world model
            action_dim: Number of actions
            horizon: Planning horizon
            num_samples: Number of trajectories to sample
            top_k: Top trajectories to keep for CEM
            device: torch device
        """
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.top_k = top_k
        self.device = device
        
        self.world_model.eval()
    
    def random_shooting(
        self,
        state: torch.Tensor,
        discount: float = 0.99
    ) -> Tuple[int, float]:
        """
        Random shooting: sample random action sequences.
        
        Args:
            state: Current state (1, state_dim)
            discount: Reward discount factor
        
        Returns:
            (best_first_action, best_return)
        """
        best_return = -float('inf')
        best_action = 0
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Sample random action sequence
                actions_idx = torch.randint(0, self.action_dim, (self.horizon,))
                actions = F.one_hot(actions_idx, num_classes=self.action_dim).float().to(self.device)
                
                # Rollout
                trajectory = self.world_model.rollout(state, actions, self.horizon)
                
                # Compute discounted return
                rewards = trajectory['rewards'].squeeze()
                discounts = torch.pow(discount, torch.arange(len(rewards), dtype=torch.float32))
                ret = (rewards * discounts).sum().item()
                
                # Track best
                if ret > best_return:
                    best_return = ret
                    best_action = actions_idx[0].item()
        
        return best_action, best_return
    
    def cem_planning(
        self,
        state: torch.Tensor,
        iterations: int = 3,
        discount: float = 0.99
    ) -> Tuple[int, float]:
        """
        Cross-Entropy Method planning: iterative refinement.
        
        Args:
            state: Current state (1, state_dim)
            iterations: Number of CEM iterations
            discount: Reward discount factor
        
        Returns:
            (best_first_action, best_return)
        """
        # Initialize action distribution (uniform)
        action_probs = torch.ones(self.horizon, self.action_dim) / self.action_dim
        
        best_action = 0
        best_return = -float('inf')
        
        with torch.no_grad():
            for it in range(iterations):
                # Sample action sequences from distribution
                trajectories = []
                
                for _ in range(self.num_samples):
                    actions_idx = torch.multinomial(action_probs, 1).squeeze()
                    actions = F.one_hot(actions_idx, num_classes=self.action_dim).float().to(self.device)
                    
                    # Rollout
                    trajectory = self.world_model.rollout(state, actions, self.horizon)
                    
                    # Compute return
                    rewards = trajectory['rewards'].squeeze()
                    discounts = torch.pow(discount, torch.arange(len(rewards), dtype=torch.float32))
                    ret = (rewards * discounts).sum().item()
                    
                    trajectories.append((actions_idx, ret))
                
                # Sort by return
                trajectories.sort(key=lambda x: x[1], reverse=True)
                
                # Update best
                best_action = trajectories[0][0][0].item()
                best_return = trajectories[0][1]
                
                # Update distribution using top-k
                top_actions = torch.stack([t[0] for t in trajectories[:self.top_k]])
                
                # Compute new probabilities
                for t in range(self.horizon):
                    action_counts = torch.bincount(top_actions[:, t], minlength=self.action_dim)
                    action_probs[t] = action_counts.float() / action_counts.sum()
        
        return best_action, best_return
    
    def mpc_step(
        self,
        state: torch.Tensor,
        method: str = 'random_shooting',
        discount: float = 0.99
    ) -> int:
        """
        Model Predictive Control: plan and execute first action.
        
        Args:
            state: Current state
            method: 'random_shooting' or 'cem'
            discount: Reward discount
        
        Returns:
            Best first action
        """
        if method == 'cem':
            action, _ = self.cem_planning(state, iterations=3, discount=discount)
        else:
            action, _ = self.random_shooting(state, discount=discount)
        
        return action


if __name__ == "__main__":
    print(f"Testing Model-Based Planning v{__version__}...")
    
    # Create world model
    from intelligence_system.models.world_model import WorldModel
    world_model = WorldModel(state_dim=4, action_dim=2, latent_dim=32, hidden_dim=128)
    
    # Create planner
    planner = ModelBasedPlanner(
        world_model,
        action_dim=2,
        horizon=10,
        num_samples=50,
        top_k=5
    )
    
    # Test random shooting
    state = torch.randn(1, 4)
    action, ret = planner.random_shooting(state)
    print(f"Random shooting: action={action}, return={ret:.3f}")
    
    # Test CEM
    action, ret = planner.cem_planning(state, iterations=3)
    print(f"CEM: action={action}, return={ret:.3f}")
    
    # Test MPC
    action = planner.mpc_step(state, method='cem')
    print(f"MPC step: action={action}")
    
    print("✅ Model-Based Planning tests OK!")
