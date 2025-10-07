"""
Agent Behavior Learner IA³ Integration
Meta-learning agent que aprende a aprender
Merge do nosso próprio projeto agent-behavior-learner-ia3
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import warnings
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DynamicNeuralNet(nn.Module):
    """Neural network que cresce/encolhe dinamicamente"""
    def __init__(self, input_size: int, output_size: int, initial_hidden: int = 64):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = [initial_hidden]
        
        self.rebuild_network()
    
    def rebuild_network(self):
        """Rebuild network com nova arquitetura"""
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def add_neurons(self, layer_idx: int, num_neurons: int):
        """Add neurons to a layer (architecture growth)"""
        if layer_idx < len(self.hidden_sizes):
            self.hidden_sizes[layer_idx] += num_neurons
            self.rebuild_network()
            logger.info(f"🧠 Added {num_neurons} neurons to layer {layer_idx}")
    
    def prune_neurons(self, layer_idx: int, num_neurons: int):
        """Remove neurons from a layer (architecture pruning)"""
        if layer_idx < len(self.hidden_sizes):
            self.hidden_sizes[layer_idx] = max(16, self.hidden_sizes[layer_idx] - num_neurons)
            self.rebuild_network()
            logger.info(f"✂️  Pruned {num_neurons} neurons from layer {layer_idx}")

class AgentBehaviorLearner:
    """
    Meta-learning agent que aprende padrões de comportamento
    Baseado em Q-learning meta com arquitetura dinâmica
    """
    def __init__(self, state_size: int, action_size: int, checkpoint_path: Path,
                 lr: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1):
        
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_path = checkpoint_path
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Dynamic Q-network
        self.q_network = DynamicNeuralNet(state_size, action_size, initial_hidden=64)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Behavior patterns detected
        self.patterns = []
        self.pattern_rewards = []
        
        # Meta-learning metrics
        self.meta_metrics = {
            'total_patterns': 0,
            'architecture_changes': 0,
            'emergent_behaviors': 0
        }
        
        # V6 PATCH: Cooldown tracking
        self.last_architecture_change_cycle = 0
        self.cycles_since_change = 0
        self.performance_history = []
        
        if checkpoint_path.exists():
            self.load()
            logger.info(f"✅ Loaded Agent Behavior Learner from {checkpoint_path}")
        else:
            logger.info("🆕 Created new Agent Behavior Learner")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy Q-learning
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, always greedy
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def learn(self, state, action, reward, next_state, done) -> float:
        """
        Q-learning update
        FIX F#4: Shape mismatch resolvido (forçar shapes [1])
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Current Q-value: FORÇAR shape [1]
        q_values = self.q_network(state_tensor)
        q_value = q_values[0, action].unsqueeze(0)  # [] → [1]
        
        # Target Q-value: JÁ shape [1]
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            next_q_value = next_q_values.max(1)[0]  # shape: [1]
            target = reward + (1 - done) * self.gamma * next_q_value  # [1]
        
        # Loss (agora OK: [1] vs [1])
        loss = self.criterion(q_value, target)
        
        # Optimize
        self.optimizer.zero_grad()
        # FIX P#5: Suppress incompletude coroutine warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*backward_with_incompletude.*")
            loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def detect_pattern(self, trajectory: List[Dict]) -> Optional[str]:
        """
        Detect behavioral patterns in trajectory
        Meta-learning capability
        """
        if len(trajectory) < 10:
            return None
        
        # Analyze reward pattern
        rewards = [t['reward'] for t in trajectory]
        avg_reward = np.mean(rewards)
        
        # Detect if new pattern emerged
        pattern_signature = f"avg_reward_{avg_reward:.2f}"
        
        if pattern_signature not in self.patterns:
            self.patterns.append(pattern_signature)
            self.pattern_rewards.append(avg_reward)
            self.meta_metrics['total_patterns'] += 1
            logger.info(f"🔍 New behavior pattern detected: {pattern_signature}")
            return pattern_signature
        
        return None
    
    def adapt_architecture(self, performance: float):
        """
        Adapt network architecture based on performance
        Meta-learning: grow if stuck, prune if overfitting
        V6 PATCH: More conservative growth with cooldown
        """
        # V6 PATCH: Track last change and require cooldown
        if not hasattr(self, 'last_architecture_change_cycle'):
            self.last_architecture_change_cycle = 0
        if not hasattr(self, 'cycles_since_change'):
            self.cycles_since_change = 0
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
        
        self.cycles_since_change += 1
        self.performance_history.append(performance)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        # V6 PATCH: Much more conservative threshold (0.15 instead of 0.3)
        # And require cooldown of 10 cycles between changes
        if performance < 0.15 and self.cycles_since_change >= 10:
            # Check if performance is truly stuck (no improvement in last 10 cycles)
            if len(self.performance_history) >= 10:
                recent_avg = sum(self.performance_history[-10:]) / 10
                older_avg = sum(self.performance_history[-20:-10]) / 10 if len(self.performance_history) >= 20 else recent_avg
                
                if recent_avg <= older_avg + 0.02:  # No real improvement
                    # Add neurons (but less: 8 instead of 16)
                    layer = np.random.randint(len(self.q_network.hidden_sizes))
                    self.q_network.add_neurons(layer, 8)  # V6 PATCH: Reduced from 16
                    self.meta_metrics['architecture_changes'] += 1
                    self.cycles_since_change = 0
                    logger.info(f"📈 Architecture grew cautiously (stuck at {performance:.2f})")
            
        elif performance > 0.95 and self.cycles_since_change >= 10:
            # Prune neurons
            layer = np.random.randint(len(self.q_network.hidden_sizes))
            self.q_network.prune_neurons(layer, 8)
            self.meta_metrics['architecture_changes'] += 1
            self.cycles_since_change = 0
            logger.info(f"✂️  Architecture pruned (high performance)")
    
    def get_meta_metrics(self) -> Dict[str, Any]:
        """Get meta-learning metrics"""
        return {
            **self.meta_metrics,
            'hidden_sizes': self.q_network.hidden_sizes,
            'total_params': sum(p.numel() for p in self.q_network.parameters()),
            'patterns_detected': len(self.patterns)
        }
    
    def save(self):
        """Save checkpoint"""
        try:
            torch.save({
                'network_state_dict': self.q_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'hidden_sizes': self.q_network.hidden_sizes,
                'patterns': self.patterns,
                'pattern_rewards': self.pattern_rewards,
                'meta_metrics': self.meta_metrics,
                'epsilon': self.epsilon
            }, self.checkpoint_path)
            logger.info(f"💾 Agent Behavior Learner saved")
        except Exception as e:
            logger.error(f"Failed to save: {e}")
    
    def load(self):
        """Load checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path)
            
            # Restore architecture first
            self.q_network.hidden_sizes = checkpoint.get('hidden_sizes', [64])
            self.q_network.rebuild_network()
            
            # Load weights
            self.q_network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore meta data
            self.patterns = checkpoint.get('patterns', [])
            self.pattern_rewards = checkpoint.get('pattern_rewards', [])
            self.meta_metrics = checkpoint.get('meta_metrics', {
                'total_patterns': 0,
                'architecture_changes': 0,
                'emergent_behaviors': 0
            })
            self.epsilon = checkpoint.get('epsilon', 0.1)
            
            logger.info(f"📂 Agent Behavior Learner loaded")
        except Exception as e:
            logger.error(f"Failed to load: {e}")
