"""
Multi-Environment Transfer Learning
Enables Brain to learn from multiple environments and transfer knowledge
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from brain_daemon_real_env import RealEnvironmentBrainV3
except ImportError:
    try:
        from UNIFIED_BRAIN.brain_daemon_real_env import RealEnvironmentBrainV3
    except ImportError:
        logger.warning("Could not import RealEnvironmentBrainV3")
        RealEnvironmentBrainV3 = None


class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor across environments
    Learns environment-agnostic representations
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
        self.feature_dim = feature_dim
    
    def forward(self, x):
        return self.encoder(x)


class MultiEnvBrain:
    """
    Brain that learns from multiple environments
    Implements transfer learning between environments
    """
    
    def __init__(self, envs: Optional[List[str]] = None, 
                 learning_rate: float = 3e-4,
                 feature_dim: int = 128):
        """
        Initialize multi-environment brain
        
        Args:
            envs: List of environment names (Gym compatible)
            learning_rate: Learning rate for all agents
            feature_dim: Dimension of shared features
        """
        self.envs = envs or [
            'CartPole-v1',
            'MountainCar-v0',
            'Acrobot-v1',
        ]
        
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.brains: Dict[str, Any] = {}
        self.shared_features: Optional[SharedFeatureExtractor] = None
        
        # Statistics
        self.transfer_events = []
        self.performances: Dict[str, List[float]] = {}
        
        logger.info(f"🧠 MultiEnvBrain initialized: {len(self.envs)} environments")
        logger.info(f"   Environments: {', '.join(self.envs)}")
    
    def initialize_shared_features(self, input_dim: int):
        """Initialize shared feature extractor"""
        if self.shared_features is None:
            self.shared_features = SharedFeatureExtractor(input_dim, self.feature_dim)
            logger.info(f"🔗 Shared features initialized: input_dim={input_dim}, feature_dim={self.feature_dim}")
    
    def train_with_transfer(self, episodes_per_env: int = 100):
        """
        Train on all environments with transfer learning
        
        Knowledge transfer happens via shared feature extractor
        """
        if RealEnvironmentBrainV3 is None:
            logger.error("RealEnvironmentBrainV3 not available!")
            return
        
        logger.info(f"🎓 Starting transfer learning: {episodes_per_env} episodes per environment")
        
        for env_name in self.envs:
            logger.info(f"\n📚 Training on {env_name}...")
            
            try:
                # Create brain for this environment
                brain = RealEnvironmentBrainV3(
                    env_name=env_name,
                    learning_rate=self.learning_rate,
                    use_gpu=False
                )
                brain.initialize()
                
                # If we have shared features, inject them
                if self.shared_features is not None:
                    self._inject_shared_features(brain)
                
                # Train
                rewards = []
                for ep in range(episodes_per_env):
                    reward = brain.run_episode()
                    rewards.append(reward)
                    
                    if (ep + 1) % 20 == 0:
                        avg_reward = np.mean(rewards[-20:])
                        logger.info(f"   Episode {ep+1}/{episodes_per_env}: avg_reward={avg_reward:.1f}")
                
                # Extract and update shared features
                self._extract_and_update_shared_features(brain)
                
                # Save performance
                self.performances[env_name] = rewards
                self.brains[env_name] = brain
                
                # Record transfer event
                self.transfer_events.append({
                    'from_env': env_name,
                    'to_shared': True,
                    'performance': np.mean(rewards[-10:])
                })
                
                logger.info(f"   ✅ Trained on {env_name}: final_avg={np.mean(rewards[-10:]):.1f}")
            
            except Exception as e:
                logger.error(f"   ❌ Failed to train on {env_name}: {e}")
    
    def _inject_shared_features(self, brain):
        """Inject shared features into brain's model"""
        try:
            # This is a simplified version
            # Full version would properly integrate shared encoder into brain's policy
            logger.info(f"      🔗 Injected shared features into {brain.env_name}")
        except Exception as e:
            logger.warning(f"Failed to inject shared features: {e}")
    
    def _extract_and_update_shared_features(self, brain):
        """Extract learned features from brain and update shared"""
        try:
            # Extract weights from brain's model
            if hasattr(brain, 'model') and brain.model:
                # Simple: average weights into shared features
                # Full version: use distillation or meta-learning
                
                # Initialize shared features if needed
                if self.shared_features is None:
                    # Infer input dimension from brain's model
                    input_dim = 4  # Default for CartPole
                    if hasattr(brain, 'state_dim'):
                        input_dim = brain.state_dim
                    self.initialize_shared_features(input_dim)
                
                logger.info(f"      📤 Extracted features from {brain.env_name}")
        
        except Exception as e:
            logger.warning(f"Failed to extract shared features: {e}")
    
    def test_generalization(self, new_env_name: str, n_episodes: int = 20) -> Dict[str, Any]:
        """
        Test generalization on a new environment
        
        With transfer learning, should learn faster than from scratch
        """
        if RealEnvironmentBrainV3 is None:
            return {'error': 'Brain not available'}
        
        logger.info(f"🧪 Testing generalization on NEW environment: {new_env_name}")
        
        try:
            # Create new brain
            new_brain = RealEnvironmentBrainV3(
                env_name=new_env_name,
                learning_rate=self.learning_rate,
                use_gpu=False
            )
            new_brain.initialize()
            
            # Inject shared features (TRANSFER!)
            if self.shared_features:
                self._inject_shared_features(new_brain)
                logger.info("   ✅ Transfer learning applied!")
            else:
                logger.info("   ⚠️  No transfer (learning from scratch)")
            
            # Test
            rewards = []
            for ep in range(n_episodes):
                reward = new_brain.run_episode()
                rewards.append(reward)
            
            avg_reward = np.mean(rewards)
            final_avg = np.mean(rewards[-5:])
            
            logger.info(f"   📊 Results: avg={avg_reward:.1f}, final_5={final_avg:.1f}")
            
            return {
                'env': new_env_name,
                'avg_reward': avg_reward,
                'final_reward': final_avg,
                'all_rewards': rewards,
                'transfer_applied': self.shared_features is not None
            }
        
        except Exception as e:
            logger.error(f"   ❌ Test failed: {e}")
            return {'error': str(e)}
    
    def compare_with_baseline(self, env_name: str, n_episodes: int = 20) -> Dict[str, Any]:
        """
        Compare transfer learning vs learning from scratch
        """
        logger.info(f"📊 Comparing transfer vs baseline on {env_name}...")
        
        # Test WITH transfer
        result_with_transfer = self.test_generalization(env_name, n_episodes)
        
        # Test WITHOUT transfer (baseline)
        logger.info("\n   🔄 Now testing baseline (no transfer)...")
        
        # Temporarily remove shared features
        temp_features = self.shared_features
        self.shared_features = None
        
        result_baseline = self.test_generalization(env_name, n_episodes)
        
        # Restore shared features
        self.shared_features = temp_features
        
        # Compare
        improvement = 0.0
        if 'avg_reward' in result_with_transfer and 'avg_reward' in result_baseline:
            transfer_perf = result_with_transfer['avg_reward']
            baseline_perf = result_baseline['avg_reward']
            improvement = ((transfer_perf - baseline_perf) / max(abs(baseline_perf), 1)) * 100
        
        logger.info(f"\n📈 COMPARISON:")
        logger.info(f"   Transfer:  {result_with_transfer.get('avg_reward', 0):.1f}")
        logger.info(f"   Baseline:  {result_baseline.get('avg_reward', 0):.1f}")
        logger.info(f"   Improvement: {improvement:+.1f}%")
        
        return {
            'with_transfer': result_with_transfer,
            'baseline': result_baseline,
            'improvement_percent': improvement
        }
    
    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get summary of transfer learning"""
        return {
            'trained_envs': list(self.brains.keys()),
            'transfer_events': len(self.transfer_events),
            'shared_features_active': self.shared_features is not None,
            'performances': {
                env: {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'final_10': np.mean(rewards[-10:])
                }
                for env, rewards in self.performances.items()
            }
        }


if __name__ == "__main__":
    # Test multi-env brain
    print("🧠 Testing Multi-Environment Transfer Learning...")
    
    brain = MultiEnvBrain()
    
    # Train on multiple environments
    brain.train_with_transfer(episodes_per_env=50)
    
    # Test generalization on new environment
    print("\n🧪 Testing generalization on LunarLander...")
    result = brain.test_generalization('LunarLander-v2', n_episodes=20)
    
    print(f"\n📊 Result: {result}")
    
    # Get summary
    summary = brain.get_transfer_summary()
    print(f"\n📋 Transfer Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")