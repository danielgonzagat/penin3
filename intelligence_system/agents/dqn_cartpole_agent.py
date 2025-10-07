#!/usr/bin/env python3
"""
DQN CartPole Agent - REAL learning with Deep Q-Network
Extracted from: teis_brutal_evolution.py
V7.0 Integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Simple but effective Q-Network"""
    
    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store experience"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch for training"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNCartPoleAgent:
    """
    DQN Agent for CartPole with REAL learning
    
    Features:
    - Double Q-Network (Q + Target)
    - Experience Replay
    - Epsilon-greedy exploration
    - Gradient clipping
    - Target network periodic update
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        model_path: Path,
        hidden_size: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        update_target_every: int = 100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_every = update_target_every
        
        # Q-Networks
        self.q_network = DQNNetwork(state_size, hidden_size, action_size)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience Replay
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Training tracking
        self.steps = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Try to load existing model
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.epsilon = checkpoint.get('epsilon', epsilon_start)
                self.steps = checkpoint.get('steps', 0)
                logger.info(f"✅ Loaded DQN model from {model_path}")
                logger.info(f"   Steps: {self.steps}, Epsilon: {self.epsilon:.3f}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load model: {e}")
        else:
            logger.info("🆕 Created new DQN agent")
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Environment state
            
        Returns:
            Action index
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Greedy exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, float(done))
    
    def update(self) -> Optional[dict]:
        """
        Update Q-network using experience replay
        
        Returns:
            Dictionary with training metrics or None if not enough samples
        """
        # Check if we have enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"🎯 Updated target network at step {self.steps}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
    
    def save(self):
        """Save model checkpoint"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, self.model_path)
        logger.debug(f"💾 Saved DQN model to {self.model_path}")


if __name__ == "__main__":
    # Quick test
    import gymnasium as gym
    
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNCartPoleAgent(
        state_size=state_size,
        action_size=action_size,
        model_path=Path("test_dqn.pth")
    )
    
    print(f"✅ DQN Agent created!")
    print(f"   State size: {state_size}")
    print(f"   Action size: {action_size}")
    print(f"   Epsilon: {agent.epsilon:.3f}")
    print(f"   Buffer: {len(agent.memory)}/{agent.memory.capacity}")
    
    # Test one episode
    state, _ = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        total_reward += reward
        
        if len(agent.memory) >= agent.batch_size:
            metrics = agent.update()
            if metrics:
                print(f"   Loss: {metrics['loss']:.4f}")
        
        if done:
            break
        
        state = next_state
    
    print(f"   Reward: {total_reward}")
    agent.save()
    print("✅ Test complete!")
