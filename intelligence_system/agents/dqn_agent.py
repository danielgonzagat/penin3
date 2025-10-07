"""
Professional DQN Agent for CartPole
REAL Reinforcement Learning, not random!
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Professional DQN Agent with proper training"""
    
    def __init__(self, state_size: int, action_size: int, model_path: Path,
                 hidden_size: int = 128, lr: float = 0.001, 
                 gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_size)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        
        # Load if exists
        if model_path.exists():
            self.load()
            logger.info(f"✅ Loaded existing DQN model from {model_path}")
        else:
            logger.info("🆕 Created new DQN agent")
        
        self.steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(batch[0])
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch[2])
        next_state_batch = torch.FloatTensor(batch[3])
        done_batch = torch.FloatTensor(batch[4])
        
        # Compute Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        
        # Update target network periodically
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self):
        """Save agent checkpoint"""
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
            }, self.model_path)
            logger.info(f"💾 DQN agent saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
    
    def load(self):
        """Load agent checkpoint"""
        try:
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            logger.info(f"📂 DQN agent loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
