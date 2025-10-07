"""
TRL Integration - COMPLETO - Transformer RL from Human Feedback
Merge REAL do /root/trl com RLHF capabilities completas
"""
import logging
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import from installed TRL
try:
    sys.path.insert(0, '/root/trl')
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl import create_reference_model
    TRL_AVAILABLE = True
    logger.info("✅ TRL imported successfully from /root/trl")
except ImportError as e:
    logger.warning(f"TRL not available: {e}")
    TRL_AVAILABLE = False

class TRLRLHFTrainer:
    """
    TRL-based RLHF trainer - PRODUCTION READY
    Reinforcement Learning from Human Feedback for LLMs
    """
    
    def __init__(self, learning_rate: float = 1.41e-5,
                 batch_size: int = 16,
                 mini_batch_size: int = 4):
        self.trl_available = TRL_AVAILABLE
        self.trainer = None
        self.model = None
        self.ref_model = None
        self.config = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.training_steps = 0
        
        logger.info(f"🎓 TRL RLHF Trainer initialized (available: {TRL_AVAILABLE})")
    
    def setup_training(self, model_name: str, 
                      reward_model: Optional[object] = None,
                      value_head: bool = True):
        """
        Setup RLHF training with full configuration
        
        Args:
            model_name: Base model name (e.g., "gpt2")
            reward_model: Optional reward model for scoring
            value_head: Add value head for PPO
        """
        if not self.trl_available:
            logger.warning("TRL not available")
            return False
        
        try:
            # Create PPO configuration
            self.config = PPOConfig(
                model_name=model_name,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                gradient_accumulation_steps=1,
                optimize_cuda_cache=True,
                early_stopping=False,
                target_kl=0.1,
                ppo_epochs=4,
                seed=42
            )
            
            # Load model with value head if needed
            if value_head:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
                logger.info(f"✅ Loaded model with value head: {model_name}")
            else:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                logger.info(f"✅ Loaded base model: {model_name}")
            
            # Create reference model
            self.ref_model = create_reference_model(self.model)
            logger.info("✅ Created reference model")
            
            # Setup reward model if provided
            self.reward_model = reward_model
            
            logger.info(f"✅ TRL RLHF fully configured for {model_name}")
            logger.info(f"   LR: {self.learning_rate}, Batch: {self.batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"TRL setup failed: {e}")
            return False
    
    def create_ppo_trainer(self, tokenizer, dataset=None):
        """
        Create PPO trainer instance
        
        Args:
            tokenizer: Tokenizer for the model
            dataset: Optional dataset for training
        """
        if not self.trl_available or self.model is None:
            logger.warning("Cannot create trainer: model not initialized")
            return False
        
        try:
            self.trainer = PPOTrainer(
                config=self.config,
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=tokenizer,
                dataset=dataset
            )
            
            logger.info("✅ PPO Trainer created successfully")
            return True
            
        except Exception as e:
            logger.error(f"PPO Trainer creation failed: {e}")
            return False
    
    def train_step(self, queries: List[str], responses: List[str], 
                   rewards: List[float]) -> Dict[str, float]:
        """
        Execute one RLHF training step with PPO
        
        Args:
            queries: Input prompts
            responses: Model generated responses
            rewards: Reward signals from reward model
        
        Returns:
            Training metrics (loss, reward_mean, etc)
        """
        if not self.trl_available or self.trainer is None:
            return {'error': 'TRL trainer not available', 'loss': 0.0}
        
        try:
            import torch
            
            # Convert to tensors
            query_tensors = [torch.tensor(q) for q in queries]
            response_tensors = [torch.tensor(r) for r in responses]
            reward_tensors = [torch.tensor(rew) for rew in rewards]
            
            # PPO step
            stats = self.trainer.step(query_tensors, response_tensors, reward_tensors)
            
            self.training_steps += 1
            
            metrics = {
                'loss': float(stats.get('ppo/loss/total', 0.0)),
                'reward_mean': float(torch.stack(reward_tensors).mean()),
                'reward_std': float(torch.stack(reward_tensors).std()),
                'ppo/policy_loss': float(stats.get('ppo/loss/policy', 0.0)),
                'ppo/value_loss': float(stats.get('ppo/loss/value', 0.0)),
                'training_steps': self.training_steps
            }
            
            logger.info(f"✅ RLHF step complete: loss={metrics['loss']:.4f}, reward={metrics['reward_mean']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"RLHF training step failed: {e}")
            return {'error': str(e), 'loss': 0.0}
    
    def save_model(self, save_path: Path):
        """Save fine-tuned model"""
        if self.model is None:
            logger.warning("No model to save")
            return False
        
        try:
            self.model.save_pretrained(str(save_path))
            logger.info(f"✅ Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive TRL statistics"""
        return {
            'trl_available': self.trl_available,
            'trainer_initialized': self.trainer is not None,
            'model_loaded': self.model is not None,
            'ref_model_loaded': self.ref_model is not None,
            'training_steps': self.training_steps,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
    
    def is_ready(self) -> bool:
        """Check if RLHF trainer is ready"""
        return self.trl_available and self.trainer is not None

