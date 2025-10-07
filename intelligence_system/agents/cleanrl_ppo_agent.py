"""
CleanRL PPO Agent Integration
Merge do cleanrl para advanced RL
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import gymnasium as gym
from collections import deque

logger = logging.getLogger(__name__)

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return shared embedding (for memory/retrieval)."""
        return self.shared(x)

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    Based on CleanRL implementation - significantly better than basic DQN
    """
    def __init__(self, state_size: int, action_size: int, model_path: Path,
                 hidden_size: int = 128, lr: float = 1e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_coef: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 batch_size: int = 64, n_steps: int = 128, n_epochs: int = 10):
        
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
        
        # Network
        self.network = PPONetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.total_steps = 0
        self.best_avg_reward = -float('inf')
        self.lr = lr  # Store for diagnostics
        # Embedding memory (for vector memory of trajectories)
        self.embedding_memory: deque = deque(maxlen=1000)
        # Episode-level embedding→return memory for retrieval warm-start
        self.embedding_reward_memory: list = []  # list of (np.ndarray, float)
        
        if model_path.exists():
            self.load()
            logger.info(f"✅ Loaded PPO agent from {model_path}")
        else:
            logger.info("🆕 Created new PPO agent (CleanRL-based)")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using policy network
        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, value = self.network(state_tensor)
            try:
                # Store embedding for vector memory (best-effort)
                emb = self.network.embed(state_tensor).squeeze(0).cpu().numpy()
                self.embedding_memory.append(emb)
            except Exception:
                pass
            
            # Sample from policy
            probs = torch.softmax(action_logits, dim=1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in trajectory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        PPO update using collected trajectory
        """
        # FIXED: Don't skip update for short trajectories!
        if len(self.states) < 1:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # Compute next value for GAE
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # FIX P#6: Convert to tensors efficiently (np.array first)
        states_tensor = torch.FloatTensor(np.array(self.states))
        actions_tensor = torch.LongTensor(self.actions)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs)
        
        # PPO update for n_epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(self.n_epochs):
            # Forward pass
            action_logits, values = self.network(states_tensor)
            
            # Policy loss
            probs = torch.softmax(action_logits, dim=1)
            action_dist = torch.distributions.Categorical(probs)
            log_probs = action_dist.log_prob(actions_tensor)
            entropy = action_dist.entropy().mean()
            
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            # FIX P#5: Suppress incompletude coroutine warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*backward_with_incompletude.*")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Extra evidence: log batch grad norm after final step
        try:
            total_norm = 0.0
            for p in self.network.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.data.norm(2).item())
            logger.info(f"[PPO] grad_norm_last_step={total_norm:.8f}")
        except Exception:
            pass

        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
        return {
            "loss": total_loss / self.n_epochs,
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs
        }

    def get_shared_embedding(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Compute and return shared embedding for a given state."""
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                emb = self.network.embed(state_tensor).squeeze(0).cpu().numpy()
                return emb
        except Exception:
            return None

    def add_episode_embedding(self, embedding: np.ndarray, episode_return: float) -> None:
        """Record (embedding, return) pair for retrieval-based warm-starts."""
        try:
            if embedding is None:
                return
            self.embedding_reward_memory.append((embedding.astype(np.float32), float(episode_return)))
            # Keep memory bounded
            if len(self.embedding_reward_memory) > 2000:
                self.embedding_reward_memory = self.embedding_reward_memory[-2000:]
        except Exception:
            pass

    def estimate_retrieval_boost(self, embedding: Optional[np.ndarray], k: int = 5) -> float:
        """Estimate expected return around an embedding using k-NN (cosine).
        Returns a boost value (neighbors_mean - global_mean)."""
        try:
            if embedding is None or not self.embedding_reward_memory:
                return 0.0
            import numpy as _np
            emb = embedding.astype(_np.float32)
            # Compute similarities
            sims = []
            for e, ret in self.embedding_reward_memory:
                # cosine similarity
                num = float(_np.dot(emb, e))
                den = float(_np.linalg.norm(emb) * _np.linalg.norm(e) + 1e-8)
                sims.append((num / den, ret))
            sims.sort(key=lambda x: x[0], reverse=True)
            top = sims[:max(1, k)]
            mean_top = float(_np.mean([r for _, r in top]))
            global_mean = float(_np.mean([r for _, r in sims])) if sims else 0.0
            return mean_top - global_mean
        except Exception:
            return 0.0
    
    def get_avg_reward(self) -> float:
        """Get average reward over last 100 episodes"""
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
    
    def save(self, force=False):
        """Save model checkpoint (only if better or forced)"""
        try:
            current_avg = self.get_avg_reward()
            
            # Save if better or forced
            if force or current_avg > self.best_avg_reward:
                self.best_avg_reward = current_avg
                
                torch.save({
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode_rewards': list(self.episode_rewards),
                    'total_steps': self.total_steps,
                    'best_avg_reward': self.best_avg_reward
                }, self.model_path)
                logger.info(f"💾 PPO agent saved (avg={current_avg:.1f}, best={self.best_avg_reward:.1f})")
            else:
                logger.debug(f"Skipping save (current={current_avg:.1f} <= best={self.best_avg_reward:.1f})")
        except Exception as e:
            logger.error(f"Failed to save PPO agent: {e}")
    
    def load(self):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
            logger.info(f"📂 PPO agent loaded (best={self.best_avg_reward:.1f})")
        except Exception as e:
            logger.error(f"Failed to load PPO agent: {e}")
