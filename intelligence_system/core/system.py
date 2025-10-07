"""
Professional Intelligence System
Clean, modular, production-ready
"""
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any
import gymnasium as gym
from collections import deque

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import *
from core.database import Database
from models.mnist_classifier import MNISTClassifier
from agents.dqn_agent import DQNAgent
from apis.api_manager import APIManager

# Ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "intelligence.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class IntelligenceSystem:
    """
    Professional AI System
    - REAL RL (DQN, not random!)
    - Saves/loads models
    - Uses APIs productively
    - Proper error handling
    - Production-ready
    """
    
    def __init__(self):
        logger.info("="*80)
        logger.info("🚀 PROFESSIONAL INTELLIGENCE SYSTEM")
        logger.info("="*80)
        
        # Core components
        self.db = Database(DATABASE_PATH)
        self.cycle = self.db.get_last_cycle()
        self.best = self.db.get_best_metrics()
        
        # ML Models
        self.mnist = MNISTClassifier(
            MNIST_MODEL_PATH,
            hidden_size=MNIST_CONFIG["hidden_size"],
            lr=MNIST_CONFIG["lr"]
        )
        
        # RL Agent (REAL DQN!)
        self.env = gym.make('CartPole-v1')
        self.dqn_agent = DQNAgent(
            state_size=4,
            action_size=2,
            model_path=DQN_MODEL_PATH,
            **DQN_CONFIG
        )
        self.cartpole_rewards = deque(maxlen=100)
        
        # APIs
        self.api_manager = APIManager(API_KEYS, API_MODELS)
        
        logger.info(f"✅ System initialized at cycle {self.cycle}")
        logger.info(f"📊 Best MNIST: {self.best['mnist']:.1f}% | Best CartPole: {self.best['cartpole']:.1f}")
    
    def run_cycle(self):
        """Execute one complete cycle"""
        self.cycle += 1
        logger.info("")
        logger.info("="*80)
        logger.info(f"🔄 CYCLE {self.cycle}")
        logger.info("="*80)
        
        # Train MNIST
        mnist_metrics = self._train_mnist()
        
        # Train CartPole with DQN
        cartpole_metrics = self._train_cartpole()
        
        # Save cycle
        self.db.save_cycle(
            self.cycle,
            mnist=mnist_metrics['test'],
            cartpole=cartpole_metrics['reward'],
            cartpole_avg=cartpole_metrics['avg_reward']
        )
        
        # Check for improvements
        self._check_records(mnist_metrics, cartpole_metrics)
        
        # Save models periodically
        if self.cycle % CHECKPOINT_INTERVAL == 0:
            self.mnist.save()
            self.dqn_agent.save()
            logger.info("💾 Models saved")
        
        # Consult APIs for improvements
        if self.cycle % API_CALL_INTERVAL == 0:
            self._consult_apis(mnist_metrics, cartpole_metrics)
        
        return mnist_metrics, cartpole_metrics
    
    def _train_mnist(self) -> Dict[str, float]:
        """Train MNIST classifier"""
        logger.info("🧠 Training MNIST...")
        
        train_acc = self.mnist.train_epoch()
        test_acc = self.mnist.evaluate()
        
        logger.info(f"   Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
        return {"train": train_acc, "test": test_acc}
    
    def _train_cartpole(self, episodes: int = 5) -> Dict[str, float]:
        """Train CartPole with DQN"""
        logger.info("🎮 Training CartPole with DQN...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # DQN action selection (epsilon-greedy)
                action = self.dqn_agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.dqn_agent.store_transition(state, action, reward, next_state, float(done))
                
                # Train DQN
                loss = self.dqn_agent.train_step()
                
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            self.cartpole_rewards.append(total_reward)
        
        avg_reward = sum(self.cartpole_rewards) / len(self.cartpole_rewards)
        last_reward = episode_rewards[-1]
        
        logger.info(f"   Last: {last_reward:.1f} | Avg(100): {avg_reward:.1f} | ε: {self.dqn_agent.epsilon:.3f}")
        
        return {"reward": last_reward, "avg_reward": avg_reward}
    
    def _check_records(self, mnist_metrics: Dict, cartpole_metrics: Dict):
        """Check and update records"""
        if mnist_metrics['test'] > self.best['mnist']:
            self.best['mnist'] = mnist_metrics['test']
            logger.info(f"   🏆 NEW MNIST RECORD: {mnist_metrics['test']:.2f}%")
        
        if cartpole_metrics['avg_reward'] > self.best['cartpole']:
            self.best['cartpole'] = cartpole_metrics['avg_reward']
            logger.info(f"   🏆 NEW CARTPOLE RECORD: {cartpole_metrics['avg_reward']:.1f}")
    
    def _consult_apis(self, mnist_metrics: Dict, cartpole_metrics: Dict):
        """Consult APIs for actionable improvements"""
        logger.info("🌐 Consulting APIs for improvements...")
        
        # Build metrics
        recent_cycles = self.db.get_recent_cycles(5)
        metrics = {
            "mnist_test": mnist_metrics['test'],
            "mnist_train": mnist_metrics['train'],
            "cartpole_avg": cartpole_metrics['avg_reward'],
            "cartpole_last": cartpole_metrics['reward'],
            "epsilon": self.dqn_agent.epsilon,
            "cycle": self.cycle,
            "stagnation": self.db.get_stagnation_score(10),
            "recent_mnist": [c['mnist_accuracy'] for c in recent_cycles if c['mnist_accuracy']],
            "recent_cartpole": [c['cartpole_avg_reward'] for c in recent_cycles if c['cartpole_avg_reward']]
        }
        
        # Get suggestions
        suggestions = self.api_manager.consult_for_improvement(metrics)
        
        # Apply suggestions (if reasonable)
        self._apply_api_suggestions(suggestions)
        
        # Save to DB
        for reasoning in suggestions.get("reasoning", []):
            self.db.save_api_response(
                self.cycle,
                reasoning["api"],
                "improvement_consultation",
                reasoning["analysis"],
                "parameter_tuning"
            )
    
    def _apply_api_suggestions(self, suggestions: Dict[str, Any]):
        """Apply API suggestions to system parameters"""
        applied = []
        
        # Learning rate adjustments
        if suggestions.get("increase_lr"):
            for param_group in self.mnist.optimizer.param_groups:
                param_group['lr'] *= 1.2
            applied.append("increased MNIST LR")
        
        if suggestions.get("decrease_lr"):
            for param_group in self.mnist.optimizer.param_groups:
                param_group['lr'] *= 0.8
            applied.append("decreased MNIST LR")
        
        # Exploration adjustments
        if suggestions.get("increase_exploration"):
            self.dqn_agent.epsilon = min(0.5, self.dqn_agent.epsilon * 1.2)
            applied.append("increased exploration")
        
        if suggestions.get("decrease_exploration"):
            self.dqn_agent.epsilon = max(0.01, self.dqn_agent.epsilon * 0.8)
            applied.append("decreased exploration")
        
        if applied:
            logger.info(f"   ✅ Applied: {', '.join(applied)}")
        else:
            logger.info(f"   ℹ️  No adjustments needed")
    
    def run_forever(self):
        """Run system indefinitely"""
        logger.info("🚀 Starting continuous operation...")
        
        try:
            while True:
                try:
                    self.run_cycle()
                    time.sleep(CYCLE_INTERVAL)
                    
                except KeyboardInterrupt:
                    raise
                    
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}", exc_info=True)
                    self.db.save_error(
                        component="system",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=str(e.__traceback__),
                        cycle=self.cycle
                    )
                    time.sleep(CYCLE_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("⏹️  Shutdown requested")
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("💾 Saving final state...")
        self.mnist.save()
        self.dqn_agent.save()
        self.env.close()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    system = IntelligenceSystem()
    system.run_forever()
