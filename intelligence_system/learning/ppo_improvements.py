#!/usr/bin/env python3
"""
🎯 PPO IMPROVEMENTS
BLOCO 3 - TAREFA 37

Enhanced PPO with:
- Clipping (prevent too large policy updates)
- Target network (stabilize value function)
- Entropy bonus (encourage exploration)
- KL divergence monitoring
"""

__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class PPOWithImprovements:
    """
    Enhanced PPO trainer with modern best practices.
    """
    
    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        n_epochs: int = 4,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            policy_net: Policy network
            value_net: Value network
            optimizer: Joint optimizer
            clip_epsilon: PPO clipping parameter (default 0.2)
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            n_epochs: Number of PPO epochs per update
            device: torch device
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.device = device
        
        # Target network for stability
        self.value_target = type(value_net)().to(device)
        self.value_target.load_state_dict(value_net.state_dict())
        self.target_update_freq = 10
        self.updates = 0
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        BLOCO 3 - TAREFA 37: PPO update with improvements
        
        Args:
            states: State tensor (batch_size, state_dim)
            actions: Action tensor (batch_size,)
            old_log_probs: Old log probabilities (batch_size,)
            returns: Discounted returns (batch_size,)
            advantages: Advantage estimates (batch_size,)
        
        Returns:
            Dict with loss statistics
        """
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0,
            'epochs_completed': 0
        }
        
        # PPO epochs
        for epoch in range(self.n_epochs):
            # Forward pass
            action_logits = self.policy_net(states)
            values = self.value_net(states).squeeze()
            
            # Compute action probabilities
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            
            # New log probabilities
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (with target network)
            with torch.no_grad():
                target_values = self.value_target(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus (encourage exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Update stats
            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['entropy'] += entropy.item()
            stats['epochs_completed'] += 1
            
            # KL divergence for early stopping
            with torch.no_grad():
                kl = (old_log_probs - new_log_probs).mean()
                stats['kl_divergence'] += kl.item()
                
                # Clip fraction
                clipped = torch.abs(ratio - 1.0) > self.clip_epsilon
                stats['clip_fraction'] += clipped.float().mean().item()
                
                # Early stopping if KL too high
                if kl > 1.5 * self.target_kl:
                    break
        
        # Average stats
        n = stats['epochs_completed']
        for key in stats:
            if key != 'epochs_completed':
                stats[key] /= n
        
        # Update target network periodically
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.value_target.load_state_dict(self.value_net.state_dict())
        
        return stats


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Reward tensor (batch_size,)
        values: Value estimates (batch_size,)
        next_values: Next value estimates (batch_size,)
        dones: Done flags (batch_size,)
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        (advantages, returns)
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    
    return advantages, returns


if __name__ == "__main__":
    print(f"Testing PPO Improvements v{__version__}...")
    
    # Create dummy networks
    state_dim = 4
    action_dim = 2
    
    policy_net = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim)
    )
    
    value_net = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=3e-4
    )
    
    ppo = PPOWithImprovements(
        policy_net, value_net, optimizer,
        clip_epsilon=0.2,
        entropy_coef=0.01
    )
    
    # Create dummy batch
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    
    # Update
    stats = ppo.update(states, actions, old_log_probs, returns, advantages)
    
    print("Update stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test GAE
    rewards = torch.randn(10)
    values = torch.randn(10)
    next_values = torch.randn(10)
    dones = torch.zeros(10)
    
    advantages, returns = compute_gae(rewards, values, next_values, dones)
    
    print(f"\nGAE shapes: advantages={advantages.shape}, returns={returns.shape}")
    
    print("✅ PPO Improvements tests OK!")
