"""
Meta-Controller: UCB Bandit for Strategy Selection

Implements an Upper Confidence Bound (UCB) bandit algorithm to dynamically
select between different search strategies (scales, mutations, etc.) based
on their performance.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ArmStatistics:
    """
    Statistics for a single bandit arm (strategy).
    
    Attributes:
        name: Name of the strategy/arm.
        n_pulls: Number of times this arm was pulled.
        total_reward: Cumulative reward.
        mean_reward: Average reward.
        variance: Variance of rewards.
    """
    name: str
    n_pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    variance: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_pulls": self.n_pulls,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "variance": self.variance,
        }


class MetaController:
    """
    UCB-based meta-controller for adaptive strategy selection.
    
    Uses Upper Confidence Bound (UCB1) algorithm to balance exploration
    and exploitation when choosing between different search strategies.
    
    Args:
        arm_names: List of strategy names.
        c: Exploration constant (higher = more exploration).
        initial_pulls: Number of initial pulls for each arm before UCB.
    """
    
    def __init__(
        self,
        arm_names: List[str],
        c: float = 1.414,  # sqrt(2) is theoretical optimum
        initial_pulls: int = 3,
    ):
        self.arm_names = arm_names
        self.c = c
        self.initial_pulls = initial_pulls
        
        # Initialize arm statistics
        self.arms: Dict[str, ArmStatistics] = {
            name: ArmStatistics(name=name) for name in arm_names
        }
        
        self.total_pulls = 0
        self.history: List[Dict[str, Any]] = []
    
    def select_arm(self) -> str:
        """
        Select an arm using UCB1 algorithm.
        
        During initial phase, pulls each arm equally.
        After that, uses UCB1 to balance exploration and exploitation.
        
        Returns:
            Name of selected arm/strategy.
        """
        # Initial phase: pull each arm at least initial_pulls times
        for arm_name, arm in self.arms.items():
            if arm.n_pulls < self.initial_pulls:
                return arm_name
        
        # UCB1 phase: compute UCB scores and select best
        ucb_scores = {}
        for arm_name, arm in self.arms.items():
            if arm.n_pulls == 0:
                # Should not happen after initial phase, but handle safely
                ucb_scores[arm_name] = float('inf')
            else:
                # UCB1 formula: mean + c * sqrt(ln(N) / n_i)
                exploration_term = self.c * np.sqrt(
                    np.log(self.total_pulls + 1) / arm.n_pulls
                )
                ucb_scores[arm_name] = arm.mean_reward + exploration_term
        
        # Select arm with highest UCB score
        best_arm = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_arm
    
    def update(self, arm_name: str, reward: float):
        """
        Update statistics after pulling an arm.
        
        Args:
            arm_name: Name of the arm that was pulled.
            reward: Reward received (fitness improvement, success rate, etc.).
        """
        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        arm = self.arms[arm_name]
        
        # Update statistics
        arm.n_pulls += 1
        arm.total_reward += reward
        
        # Update mean
        old_mean = arm.mean_reward
        arm.mean_reward = arm.total_reward / arm.n_pulls
        
        # Update variance (Welford's online algorithm)
        if arm.n_pulls == 1:
            arm.variance = 0.0
        else:
            arm.variance = (
                (arm.n_pulls - 2) * arm.variance + 
                (reward - old_mean) * (reward - arm.mean_reward)
            ) / (arm.n_pulls - 1)
        
        self.total_pulls += 1
        
        # Record history
        self.history.append({
            "pull": self.total_pulls,
            "arm": arm_name,
            "reward": reward,
            "mean_reward": arm.mean_reward,
        })
    
    def get_arm_statistics(self, arm_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific arm.
        
        Args:
            arm_name: Name of the arm.
            
        Returns:
            Dictionary with arm statistics.
        """
        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        return self.arms[arm_name].to_dict()
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all arms.
        
        Returns:
            Dictionary with all arm statistics and summary.
        """
        arms_data = {name: arm.to_dict() for name, arm in self.arms.items()}
        
        # Summary statistics
        if self.total_pulls > 0:
            best_arm = max(
                self.arms.items(),
                key=lambda x: x[1].mean_reward
            )[0]
            most_pulled_arm = max(
                self.arms.items(),
                key=lambda x: x[1].n_pulls
            )[0]
        else:
            best_arm = None
            most_pulled_arm = None
        
        return {
            "arms": arms_data,
            "total_pulls": self.total_pulls,
            "best_arm": best_arm,
            "most_pulled_arm": most_pulled_arm,
        }
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        Get current recommendation for strategy selection.
        
        Returns:
            Dictionary with recommended arm and confidence.
        """
        if self.total_pulls == 0:
            return {
                "recommended_arm": self.arm_names[0],
                "confidence": 0.0,
                "reason": "No data yet",
            }
        
        # Find best arm by mean reward
        best_arm = max(self.arms.items(), key=lambda x: x[1].mean_reward)
        
        # Compute confidence based on sample size and variance
        n = best_arm[1].n_pulls
        var = best_arm[1].variance
        
        if n > 0 and var >= 0:
            # Confidence interval width (smaller = more confident)
            ci_width = 1.96 * np.sqrt(var / n)  # 95% CI
            confidence = max(0.0, 1.0 - ci_width)
        else:
            confidence = 0.5
        
        return {
            "recommended_arm": best_arm[0],
            "mean_reward": best_arm[1].mean_reward,
            "n_pulls": best_arm[1].n_pulls,
            "confidence": confidence,
        }
    
    def reset(self):
        """Reset all statistics."""
        for arm in self.arms.values():
            arm.n_pulls = 0
            arm.total_reward = 0.0
            arm.mean_reward = 0.0
            arm.variance = 0.0
        self.total_pulls = 0
        self.history.clear()
