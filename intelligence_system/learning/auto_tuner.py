#!/usr/bin/env python3
"""
🎛️ AUTO-TUNER
BLOCO 3 - TAREFA 31-32

Automatically tunes hyperparameters based on performance.
Adapts learning rate, entropy, batch size, etc.
"""

__version__ = "1.0.0"

import numpy as np
from typing import Dict, Any
from collections import deque


class AutoTuner:
    """
    Adaptive hyperparameter tuning based on performance metrics.
    Uses simple heuristics to adjust parameters in real-time.
    """
    
    def __init__(self):
        """Initialize auto-tuner with default ranges"""
        self.param_ranges = {
            'lr': (1e-5, 1e-2),
            'entropy_coef': (0.0, 0.1),
            'batch_size': (16, 256),
            'gamma': (0.95, 0.999),
            'epsilon': (0.01, 0.3),
            'top_k': (2, 16),
            'temperature': (0.1, 2.0)
        }
        
        self.performance_history = deque(maxlen=100)
        self.adjustments_made = 0
    
    def tune(
        self,
        current_params: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        BLOCO 3 - TAREFA 32: Adaptive hyperparameter tuning
        
        Tune parameters based on performance.
        
        Args:
            current_params: Current hyperparameters
            performance_metrics: {
                'avg_reward': float,
                'loss': float,
                'entropy': float,
                'improvement': float  # Change from last check
            }
        
        Returns:
            Updated parameters dict
        """
        new_params = current_params.copy()
        
        # Store performance
        self.performance_history.append(performance_metrics.get('avg_reward', 0))
        
        # Need at least 10 episodes to tune
        if len(self.performance_history) < 10:
            return new_params
        
        # Get recent performance
        recent = list(self.performance_history)[-20:]
        avg_recent = np.mean(recent)
        trend = np.mean(np.diff(recent)) if len(recent) > 1 else 0
        
        improvement = performance_metrics.get('improvement', 0)
        loss = performance_metrics.get('loss', 0)
        entropy = performance_metrics.get('entropy', 0)
        
        # Tune learning rate
        if 'lr' in current_params:
            lr = current_params['lr']
            
            # If stagnating, increase LR
            if abs(trend) < 0.1 and improvement < 0.01:
                new_lr = min(lr * 1.2, self.param_ranges['lr'][1])
                new_params['lr'] = new_lr
                self.adjustments_made += 1
            
            # If unstable (high loss), decrease LR
            elif loss > 1.0:
                new_lr = max(lr * 0.8, self.param_ranges['lr'][0])
                new_params['lr'] = new_lr
                self.adjustments_made += 1
            
            # If improving well, keep it
            elif improvement > 0.05:
                pass  # Don't change
        
        # Tune entropy coefficient
        if 'entropy_coef' in current_params:
            ent_coef = current_params['entropy_coef']
            
            # If entropy too low (deterministic), increase exploration
            if entropy < 0.1:
                new_ent = min(ent_coef * 1.5, self.param_ranges['entropy_coef'][1])
                new_params['entropy_coef'] = new_ent
                self.adjustments_made += 1
            
            # If entropy too high (random), decrease exploration
            elif entropy > 0.8:
                new_ent = max(ent_coef * 0.7, self.param_ranges['entropy_coef'][0])
                new_params['entropy_coef'] = new_ent
                self.adjustments_made += 1
        
        # Tune batch size
        if 'batch_size' in current_params:
            batch_size = current_params['batch_size']
            
            # If unstable, increase batch size
            if loss > 1.0:
                new_batch = min(int(batch_size * 1.5), self.param_ranges['batch_size'][1])
                new_params['batch_size'] = new_batch
                self.adjustments_made += 1
            
            # If stable but slow, decrease batch size
            elif loss < 0.5 and improvement < 0.01:
                new_batch = max(int(batch_size * 0.8), self.param_ranges['batch_size'][0])
                new_params['batch_size'] = new_batch
                self.adjustments_made += 1
        
        # Tune router parameters (if available)
        if 'top_k' in current_params:
            top_k = current_params['top_k']
            
            # If stagnating, increase diversity
            if abs(trend) < 0.1:
                new_top_k = min(top_k + 2, self.param_ranges['top_k'][1])
                new_params['top_k'] = int(new_top_k)
                self.adjustments_made += 1
        
        if 'temperature' in current_params:
            temp = current_params['temperature']
            
            # If stagnating, increase temperature (more exploration)
            if abs(trend) < 0.1:
                new_temp = min(temp * 1.2, self.param_ranges['temperature'][1])
                new_params['temperature'] = new_temp
                self.adjustments_made += 1
            
            # If improving, decrease temperature (more exploitation)
            elif improvement > 0.1:
                new_temp = max(temp * 0.9, self.param_ranges['temperature'][0])
                new_params['temperature'] = new_temp
                self.adjustments_made += 1
        
        return new_params
    
    def get_stats(self) -> Dict:
        """Get tuning statistics"""
        if len(self.performance_history) == 0:
            return {
                'samples': 0,
                'adjustments': self.adjustments_made
            }
        
        history = list(self.performance_history)
        
        return {
            'samples': len(history),
            'avg_performance': np.mean(history),
            'std_performance': np.std(history),
            'trend': np.mean(np.diff(history)) if len(history) > 1 else 0,
            'adjustments': self.adjustments_made
        }


if __name__ == "__main__":
    print(f"Testing Auto-Tuner v{__version__}...")
    
    tuner = AutoTuner()
    
    # Simulate training with auto-tuning
    params = {
        'lr': 1e-3,
        'entropy_coef': 0.01,
        'batch_size': 64,
        'top_k': 8,
        'temperature': 1.0
    }
    
    print("Initial params:", params)
    
    # Simulate 50 episodes with varying performance
    for ep in range(50):
        # Simulate metrics
        metrics = {
            'avg_reward': 10 + ep * 0.5 + np.random.randn() * 2,  # Growing trend
            'loss': max(0.1, 1.0 - ep * 0.01),  # Decreasing loss
            'entropy': 0.5 + np.random.randn() * 0.1,
            'improvement': 0.5 if ep > 0 else 0
        }
        
        # Tune
        params = tuner.tune(params, metrics)
        
        if ep % 10 == 0:
            print(f"\nEpisode {ep}:")
            print(f"  Metrics: {metrics}")
            print(f"  Updated params: {params}")
            print(f"  Stats: {tuner.get_stats()}")
    
    print(f"\n✅ Auto-Tuner test OK!")
    print(f"Final stats: {tuner.get_stats()}")
