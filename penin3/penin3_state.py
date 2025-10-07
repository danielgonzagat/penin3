"""
PENIN³ Unified State
====================

Estado unificado combinando V7 operational e PENIN-Ω meta layers.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json

UTC = timezone.utc


@dataclass
class V7State:
    """V7 operational state"""
    cycle: int = 0
    mnist_accuracy: float = 0.0
    cartpole_reward: float = 0.0
    best_mnist: float = 0.0
    best_cartpole: float = 0.0
    total_episodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeninOmegaState:
    """PENIN-Ω meta state"""
    master_I: float = 0.0
    linf_score: float = 0.0
    caos_factor: float = 1.0
    sr_score: float = 0.0
    sigma_valid: bool = True
    stagnation_count: int = 0
    champion_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PENIN3State:
    """
    PENIN³ unified state
    
    Combines V7 operational metrics with PENIN-Ω meta-layer state.
    """
    
    # Identification
    cycle: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # V7 operational
    v7: V7State = field(default_factory=V7State)
    
    # PENIN-Ω meta
    penin_omega: PeninOmegaState = field(default_factory=PeninOmegaState)
    
    # History
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export complete state"""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp.isoformat(),
            "v7": self.v7.to_dict(),
            "penin_omega": self.penin_omega.to_dict(),
            "unified_score": self.compute_unified_score(),
            "history_length": len(self.history)
        }
    
    def to_json(self) -> str:
        """Export as JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def compute_unified_score(self) -> float:
        """
        Compute unified PENIN³ score
        
        Combines V7 operational performance with PENIN-Ω meta quality.
        
        CRITICAL FIX: Use BEST performance, not current (RL exploration tolerance)
        """
        # CRITICAL FIX: Use BEST scores for stable metrics
        best_mnist = max(self.v7.best_mnist, self.v7.mnist_accuracy)
        best_cartpole = max(self.v7.best_cartpole, self.v7.cartpole_reward)
        
        # Normalize V7 metrics (0-1)
        mnist_norm = best_mnist / 100.0
        cartpole_norm = min(best_cartpole / 500.0, 1.0)
        
        # V7 operational score (average)
        v7_score = (mnist_norm + cartpole_norm) / 2.0
        
        # PENIN-Ω meta score (L∞ already uses best)
        penin_score = self.penin_omega.linf_score
        
        # Unified score (weighted average)
        # 60% operational, 40% meta-quality
        unified = 0.6 * v7_score + 0.4 * penin_score
        
        return unified
    
    def add_to_history(self) -> None:
        """Add current state to history"""
        self.history.append({
            "cycle": self.cycle,
            "timestamp": datetime.now(UTC).isoformat(),
            "v7_mnist": self.v7.mnist_accuracy,
            "v7_cartpole": self.v7.cartpole_reward,
            "penin_master_I": self.penin_omega.master_I,
            "penin_linf": self.penin_omega.linf_score,
            "unified_score": self.compute_unified_score()
        })
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def detect_stagnation(self, window: int = 5) -> bool:
        """
        Detect if system is stagnating
        
        Returns True if no improvement in last `window` cycles.
        """
        if len(self.history) < window:
            return False
        
        recent = self.history[-window:]
        scores = [h["unified_score"] for h in recent]
        
        # Check if scores are essentially flat (< 1% variation)
        if len(scores) < 2:
            return False
        
        max_score = max(scores)
        min_score = min(scores)
        variation = (max_score - min_score) / max(max_score, 0.01)
        
        return variation < 0.01
    
    def get_improvement_rate(self, window: int = 10) -> float:
        """
        Get improvement rate over last `window` cycles
        
        Returns percentage improvement per cycle.
        """
        if len(self.history) < 2:
            return 0.0
        
        recent = self.history[-min(window, len(self.history)):]
        if len(recent) < 2:
            return 0.0
        
        first_score = recent[0]["unified_score"]
        last_score = recent[-1]["unified_score"]
        
        if first_score == 0:
            return 0.0
        
        improvement = (last_score - first_score) / first_score
        rate = improvement / len(recent)
        
        return rate * 100.0  # Percentage
    
    def save_checkpoint(self, path: str) -> None:
        """Save state checkpoint"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'PENIN3State':
        """Load state checkpoint"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
