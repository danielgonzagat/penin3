"""
NextGen Gödelian Incompleteness Integration
Anti-stagnation system que força inovação
Merge do nosso projeto nextgen-godelian-incompleteness
"""
import logging
import numpy as np
from typing import Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

class GodelianAntiStagnation:
    """
    Sistema anti-stagnation baseado em incompletude de Gödel
    Previne que o sistema fique preso em mínimos locais
    """
    def __init__(self, stagnation_threshold: float = 0.01, window_size: int = 10):
        self.threshold = stagnation_threshold
        self.window_size = window_size
        
        # Tracking
        self.metric_history = deque(maxlen=window_size)
        self.delta_0 = 0.1  # Perturbation strength
        
        # Anti-stagnation strategies
        self.strategies = [
            'increase_lr',
            'decrease_lr',
            'add_noise',
            'architecture_change',
            'data_augmentation',
            'reset_optimizer',
            'curriculum_shift'
        ]
        
        self.strategy_scores = {s: 1.0 for s in self.strategies}
        self.activations = 0
        
        logger.info("🔄 Gödelian Anti-Stagnation initialized")
    
    def is_stagnant(self, current_metric: float) -> bool:
        """
        Detect if system is stagnant
        
        Uses variance in recent metrics to determine stagnation
        """
        self.metric_history.append(current_metric)
        
        if len(self.metric_history) < self.window_size:
            return False
        
        # Calculate variance
        metrics = list(self.metric_history)
        variance = np.var(metrics)
        
        stagnant = variance < self.threshold
        
        if stagnant:
            logger.warning(f"⚠️  STAGNATION DETECTED! Variance: {variance:.6f}")
        
        return stagnant
    
    def get_antistagnation_action(self) -> Dict[str, Any]:
        """
        Get action to break stagnation
        
        Uses evolutionary strategy selection
        """
        # Select strategy based on scores (exploitation + exploration)
        strategy_probs = np.array([self.strategy_scores[s] for s in self.strategies])
        strategy_probs = strategy_probs / strategy_probs.sum()
        
        selected_strategy = np.random.choice(self.strategies, p=strategy_probs)
        
        action = {
            'strategy': selected_strategy,
            'delta_0': self.delta_0,
            'activations': self.activations
        }
        
        self.activations += 1
        self._adapt_delta()
        
        logger.info(f"🎯 Anti-stagnation: {selected_strategy} (δ₀={self.delta_0:.4f})")
        
        return action
    
    def update_strategy_score(self, strategy: str, improvement: float):
        """
        Update strategy score based on effectiveness
        Meta-learning sobre as próprias estratégias
        """
        # Exponential moving average
        alpha = 0.3
        self.strategy_scores[strategy] = (
            alpha * improvement + (1 - alpha) * self.strategy_scores[strategy]
        )
        
        logger.debug(f"📊 Strategy '{strategy}' score: {self.strategy_scores[strategy]:.3f}")
    
    def _adapt_delta(self):
        """
        Adapt perturbation strength based on activations
        Too many activations = reduce delta (being too aggressive)
        """
        if self.activations % 5 == 0:
            self.delta_0 *= 0.9  # Decay
            self.delta_0 = max(0.01, self.delta_0)  # Floor
            logger.debug(f"🔧 Adapted δ₀ to {self.delta_0:.4f}")
    
    def get_synergistic_actions(self, current_metrics: Dict[str, float]) -> List[str]:
        """
        Generate synergistic action combinations
        Advanced anti-stagnation through multi-action
        """
        actions = []
        
        mnist = current_metrics.get('mnist_test', 0)
        cartpole = current_metrics.get('cartpole_avg', 0)
        
        # If both are low, try aggressive changes
        if mnist < 50 and cartpole < 100:
            actions.extend(['increase_lr', 'architecture_change', 'add_noise'])
        
        # If one is high but other low, focus on weak one
        elif mnist > 80 and cartpole < 100:
            actions.extend(['curriculum_shift', 'increase_exploration'])
        elif cartpole > 200 and mnist < 50:
            actions.extend(['data_augmentation', 'architecture_change'])
        
        # If both high but stagnant, try subtle changes
        elif mnist > 80 and cartpole > 200:
            actions.extend(['decrease_lr', 'reset_optimizer'])
        
        logger.info(f"🎯 Synergistic actions: {actions}")
        return actions
    
    def predict_stagnation(self, metrics_sequence: List[float]) -> float:
        """
        Predict probability of future stagnation
        
        Proactive anti-stagnation
        """
        if len(metrics_sequence) < 5:
            return 0.0
        
        # Calculate trend
        recent = metrics_sequence[-5:]
        trend = np.polyfit(range(5), recent, 1)[0]
        
        # If trend is flat or negative, high stagnation risk
        if abs(trend) < 0.01:
            return 0.8
        elif trend < 0:
            return 0.9
        else:
            return max(0.0, 0.5 - trend)
    
    def get_status(self) -> Dict[str, Any]:
        """Get anti-stagnation system status"""
        return {
            'activations': self.activations,
            'delta_0': self.delta_0,
            'window_size': self.window_size,
            'threshold': self.threshold,
            'metric_history': list(self.metric_history),
            'top_strategies': sorted(
                self.strategy_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
