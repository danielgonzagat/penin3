#!/usr/bin/env python3
"""
🤺 SELF-PLAY TRAINING
BLOCO 3 - TAREFA 39

Basic self-play for competitive environments.
Agent plays against past versions of itself.
"""

__version__ = "1.0.0"

import torch
import copy
from collections import deque
from typing import List, Optional


class SelfPlayManager:
    """
    BLOCO 3 - TAREFA 39: Basic self-play
    
    Manages opponent pool for self-play training.
    Stores past versions of agent as opponents.
    """
    
    def __init__(
        self,
        max_opponents: int = 10,
        win_rate_threshold: float = 0.7,
        update_frequency: int = 50
    ):
        """
        Args:
            max_opponents: Maximum opponents to keep in pool
            win_rate_threshold: Win rate to add to opponent pool
            update_frequency: Episodes between opponent updates
        """
        self.max_opponents = max_opponents
        self.win_rate_threshold = win_rate_threshold
        self.update_frequency = update_frequency
        
        self.opponent_pool = deque(maxlen=max_opponents)
        self.current_opponent_idx = 0
        
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def add_opponent(self, model: torch.nn.Module):
        """
        Add model to opponent pool.
        
        Args:
            model: Model to add (will be deep copied)
        """
        opponent_copy = copy.deepcopy(model)
        opponent_copy.eval()  # Set to eval mode
        
        self.opponent_pool.append({
            'model': opponent_copy,
            'episode': self.episodes,
            'win_rate_vs': 0.0
        })
    
    def get_opponent(self) -> Optional[torch.nn.Module]:
        """
        Get next opponent from pool.
        
        Returns:
            Opponent model or None if pool empty
        """
        if not self.opponent_pool:
            return None
        
        opponent = self.opponent_pool[self.current_opponent_idx]
        
        # Rotate through opponents
        self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponent_pool)
        
        return opponent['model']
    
    def record_result(self, win: bool, draw: bool = False):
        """
        Record game result.
        
        Args:
            win: True if agent won
            draw: True if draw
        """
        self.episodes += 1
        
        if draw:
            self.draws += 1
        elif win:
            self.wins += 1
        else:
            self.losses += 1
    
    def should_add_to_pool(self, model: torch.nn.Module) -> bool:
        """
        Check if current model should be added to opponent pool.
        
        Args:
            model: Current model
        
        Returns:
            True if should add
        """
        # Need minimum episodes
        if self.episodes < self.update_frequency:
            return False
        
        # Check if it's time to update
        if self.episodes % self.update_frequency != 0:
            return False
        
        # Check win rate
        total_games = self.wins + self.losses + self.draws
        if total_games == 0:
            return False
        
        win_rate = self.wins / total_games
        
        if win_rate >= self.win_rate_threshold:
            self.add_opponent(model)
            
            # Reset stats
            self.wins = 0
            self.losses = 0
            self.draws = 0
            
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get self-play statistics"""
        total = self.wins + self.losses + self.draws
        
        if total == 0:
            return {
                'episodes': self.episodes,
                'opponents': len(self.opponent_pool),
                'win_rate': 0.0,
                'wins': 0,
                'losses': 0,
                'draws': 0
            }
        
        return {
            'episodes': self.episodes,
            'opponents': len(self.opponent_pool),
            'win_rate': self.wins / total,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws
        }


class CompetitiveEnvironmentWrapper:
    """
    Wrapper for competitive 2-player environments.
    Handles opponent selection and game results.
    """
    
    def __init__(self, env, self_play_manager: SelfPlayManager):
        self.env = env
        self.self_play = self_play_manager
        self.opponent = None
    
    def reset(self):
        """Reset environment and select opponent"""
        state = self.env.reset()
        self.opponent = self.self_play.get_opponent()
        return state
    
    def step(self, agent_action):
        """
        Step with both agent and opponent actions.
        
        Args:
            agent_action: Agent's action
        
        Returns:
            (state, reward, done, info)
        """
        # Get opponent action
        if self.opponent is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.env.state)
                opponent_action_logits = self.opponent(state_tensor.unsqueeze(0))
                opponent_action = torch.argmax(opponent_action_logits).item()
        else:
            # Random opponent if no pool yet
            opponent_action = self.env.action_space.sample()
        
        # Take actions (implementation depends on environment)
        # This is a simplified example
        state, reward, done, info = self.env.step(agent_action)
        
        # Record result if episode ended
        if done:
            win = reward > 0
            draw = reward == 0
            self.self_play.record_result(win, draw)
        
        return state, reward, done, info


if __name__ == "__main__":
    print(f"Testing Self-Play Manager v{__version__}...")
    
    # Create manager
    sp = SelfPlayManager(max_opponents=5, win_rate_threshold=0.7, update_frequency=10)
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )
    
    # Simulate self-play
    for ep in range(50):
        # Simulate game
        win = ep % 3 == 0  # Win 33% of time
        sp.record_result(win)
        
        # Check if should add to pool
        if sp.should_add_to_pool(model):
            print(f"Episode {ep}: Added to opponent pool")
            print(f"  Stats: {sp.get_stats()}")
    
    print(f"\nFinal stats: {sp.get_stats()}")
    print(f"Opponent pool size: {len(sp.opponent_pool)}")
    
    print("✅ Self-Play tests OK!")
