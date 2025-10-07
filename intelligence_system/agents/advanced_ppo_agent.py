"""
ADVANCED PPO AGENT - Resolver CartPole estagnado
Implementa técnicas avançadas de RL para melhorar performance
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import deque
import random

logger = logging.getLogger(__name__)


class AdvancedPPONetwork(nn.Module):
    """
    Rede neural melhorada para PPO
    - Maior capacidade
    - Batch normalization
    - Dropout para regularização
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Shared feature extraction (maior, sem BatchNorm para evitar issues com batch_size=1)
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle single state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        features = self.features(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value


class AdvancedPPOAgent:
    """
    PPO Agent com técnicas avançadas:
    - Larger replay buffer
    - Learning rate scheduling
    - Advantage normalization
    - Entropy bonus scheduling
    - Multiple update epochs
    - Gradient clipping
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        model_path: Path,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 128,  # Aumentado
        n_steps: int = 256,     # Aumentado
        n_epochs: int = 10,     # Aumentado
        max_grad_norm: float = 0.5
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network
        self.network = AdvancedPPONetwork(state_size, action_size, hidden_size).to(self.device)
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # Buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Metrics
        self.update_count = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)  # Compatibility with V6
        
        # Load if exists
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.network.load_state_dict(checkpoint['network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.update_count = checkpoint.get('update_count', 0)
                self.total_steps = checkpoint.get('total_steps', 0)
                logger.info(f"✅ Loaded Advanced PPO from {model_path}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load model: {e}")
        
        logger.info(f"🚀 Advanced PPO Agent initialized")
        logger.info(f"   Hidden size: {hidden_size}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Steps per update: {n_steps}")
        logger.info(f"   Epochs per update: {n_epochs}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        if training:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Greedy
            action = action_probs.argmax()
            log_prob = torch.log(action_probs[0, action])
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_steps += 1
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        Update policy using collected experience
        With advanced techniques
        """
        if len(self.states) < self.batch_size:
            return {'status': 'not_enough_data'}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Get final value for bootstrap
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.squeeze()
        
        # Compute advantages using GAE
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = old_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[t])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages (IMPORTANT!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'clip_fraction': 0
        }
        
        for epoch in range(self.n_epochs):
            # Mini-batch updates
            indices = list(range(len(states)))
            random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(batch_states)
                
                # Policy loss with clipping
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = values.squeeze()
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                
                # Clip fraction
                clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                metrics['clip_fraction'] += clip_fraction.item()
        
        # Average metrics
        n_updates = self.n_epochs * (len(states) // self.batch_size)
        for key in metrics:
            metrics[key] /= n_updates
        
        # Learning rate scheduling
        self.scheduler.step()
        metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        self.update_count += 1
        
        logger.info(f"   PPO Update #{self.update_count}:")
        logger.info(f"      Policy loss: {metrics['policy_loss']:.4f}")
        logger.info(f"      Value loss: {metrics['value_loss']:.4f}")
        logger.info(f"      Entropy: {metrics['entropy']:.4f}")
        logger.info(f"      LR: {metrics['learning_rate']:.6f}")
        
        return metrics
    
    def save(self):
        """Save model checkpoint"""
        checkpoint = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, self.model_path)
        logger.info(f"💾 Saved Advanced PPO to {self.model_path}")


if __name__ == "__main__":
    # Test
    agent = AdvancedPPOAgent(
        state_size=4,
        action_size=2,
        model_path=Path("test_advanced_ppo.pth")
    )
    
    print("✅ Advanced PPO Agent created successfully")
    print(f"   Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
