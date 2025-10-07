#!/usr/bin/env python3
"""
💾 EXPERIENCE REPLAY BUFFER
BLOCO 3 - TAREFA 33

Stores experiences for off-policy learning.
Improves sample efficiency significantly.
"""

__version__ = "1.0.0"

import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional
import random


class ExperienceReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    Enables off-policy learning and breaks temporal correlations.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # For prioritized replay
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
            priority: Priority for sampling (default 1.0)
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            or None if buffer too small
        """
        if len(self.buffer) < batch_size:
            return None
        
        experiences = random.sample(list(self.buffer), batch_size)
        
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def sample_tensors(
        self, 
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Optional[Tuple]:
        """
        Sample batch and convert to PyTorch tensors.
        
        Returns:
            Tuple of tensors or None
        """
        batch = self.sample(batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()
        self.priorities.clear()
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0
            }
        
        rewards = [e[2] for e in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'usage': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """
    BLOCO 3 - TAREFA 35: Prioritized Experience Replay
    
    Samples important experiences more frequently.
    Uses TD-error as priority.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ):
        """Add experience with priority"""
        if priority is None:
            priority = self.max_priority
        
        super().push(state, action, reward, next_state, done, priority)
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        Sample batch using priorities.
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities), dtype=np.float32)
        priorities = priorities ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(priority)
                self.max_priority = max(self.max_priority, float(priority))


if __name__ == "__main__":
    print(f"Testing Experience Replay Buffer v{__version__}...")
    
    # Test basic replay
    buffer = ExperienceReplayBuffer(capacity=1000)
    
    # Add some experiences
    for i in range(100):
        state = np.random.randn(4)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = np.random.randn(4)
        done = random.random() < 0.1
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")
    
    # Sample batch
    batch = buffer.sample(32)
    if batch:
        print(f"Sampled batch shapes:")
        for i, name in enumerate(['states', 'actions', 'rewards', 'next_states', 'dones']):
            print(f"  {name}: {batch[i].shape}")
    
    # Test prioritized replay
    print("\nTesting Prioritized Replay...")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(4)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = np.random.randn(4)
        done = random.random() < 0.1
        priority = random.random() * 10  # Random priorities
        
        pri_buffer.push(state, action, reward, next_state, done, priority)
    
    batch = pri_buffer.sample(32)
    if batch:
        print(f"Prioritized batch has {len(batch)} elements (includes indices and weights)")
    
    print("✅ Experience Replay tests OK!")
