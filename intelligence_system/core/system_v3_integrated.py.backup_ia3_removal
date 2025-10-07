"""
Intelligence System V3.0 - MEGA-MERGE EDITION
Integração completa de todas as tecnologias valiosas
"""
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import gymnasium as gym
from collections import deque
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import *
from core.database import Database
from models.mnist_classifier import MNISTClassifier

# Try advanced imports
try:
    from agents.cleanrl_ppo_agent import PPOAgent
    PPO_AVAILABLE = True
except:
    from agents.dqn_agent import DQNAgent
    PPO_AVAILABLE = False

try:
    from apis.litellm_wrapper import LiteLLMWrapper
    LITELLM_AVAILABLE = True
except:
    from apis.api_manager import APIManager
    LITELLM_AVAILABLE = False

try:
    from meta.agent_behavior_learner import AgentBehaviorLearner
    META_LEARNER_AVAILABLE = True
except:
    META_LEARNER_AVAILABLE = False

try:
    from meta.godelian_antistagnation import GodelianAntiStagnation
    GODELIAN_AVAILABLE = True
except:
    GODELIAN_AVAILABLE = False

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "intelligence_v3.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class IntelligenceSystemV3:
    """
    🚀 INTELLIGENCE SYSTEM V3.0 - MEGA-MERGE EDITION
    
    INTEGRATIONS:
    ✅ LiteLLM - Universal API gateway (6 APIs)
    ✅ CleanRL - Advanced RL (PPO > DQN)
    ✅ Agent Behavior Learner - Meta-learning
    ✅ Gödelian Anti-Stagnation - Force innovation
    ✅ AutoKeras - AutoML (future)
    ✅ DSPy - Meta-prompting (future)
    ✅ LangGraph - Orchestration (future)
    
    GOALS:
    - IA³ Score: 8.4% → 25%+ (first merge)
    - MNIST: 10% → 95%+
    - CartPole: 15 → 300+
    - APIs: 2 calls → 12+ calls per cycle
    """
    
    def __init__(self):
        logger.info("="*80)
        logger.info("🚀 INTELLIGENCE SYSTEM V3.0 - MEGA-MERGE EDITION")
        logger.info("="*80)
        logger.info(f"   PPO: {PPO_AVAILABLE}")
        logger.info(f"   LiteLLM: {LITELLM_AVAILABLE}")
        logger.info(f"   Meta-Learner: {META_LEARNER_AVAILABLE}")
        logger.info(f"   Gödelian: {GODELIAN_AVAILABLE}")
        logger.info("="*80)
        
        # Core
        self.db = Database(DATABASE_PATH)
        self.cycle = self.db.get_last_cycle()
        self.best = self.db.get_best_metrics()
        
        # MNIST (with future AutoKeras upgrade path)
        self.mnist = MNISTClassifier(
            MNIST_MODEL_PATH,
            hidden_size=MNIST_CONFIG["hidden_size"],
            lr=MNIST_CONFIG["lr"]
        )
        
        # RL Agent (PPO if available, else DQN)
        self.env = gym.make('CartPole-v1')
        
        if PPO_AVAILABLE:
            # PPO config (CleanRL advanced)
            self.rl_agent = PPOAgent(
                state_size=4,
                action_size=2,
                model_path=MODELS_DIR / "ppo_cartpole.pth",
                **PPO_CONFIG
            )
            logger.info("✅ Using CleanRL PPO agent")
        else:
            from agents.dqn_agent import DQNAgent
            self.rl_agent = DQNAgent(
                state_size=4,
                action_size=2,
                model_path=DQN_MODEL_PATH,
                **DQN_CONFIG
            )
            logger.info("⚠️  Using fallback DQN agent")
        
        self.cartpole_rewards = deque(maxlen=100)
        
        # APIs (LiteLLM if available, else fallback)
        if LITELLM_AVAILABLE:
            self.api_manager = LiteLLMWrapper(API_KEYS, API_MODELS)
            logger.info("✅ Using LiteLLM API wrapper (6 APIs unified)")
        else:
            from apis.api_manager import APIManager
            self.api_manager = APIManager(API_KEYS, API_MODELS)
            logger.info("⚠️  Using fallback API manager")
        
        # Meta-Learning (Agent Behavior Learner)
        if META_LEARNER_AVAILABLE:
            self.meta_learner = AgentBehaviorLearner(
                state_size=10,  # Meta-state: recent metrics
                action_size=5,  # Meta-actions: lr, arch, etc
                checkpoint_path=MODELS_DIR / "meta_learner.pth",
                lr=0.001
            )
            logger.info("✅ Meta-learner initialized")
        else:
            self.meta_learner = None
        
        # Anti-Stagnation (Gödelian)
        if GODELIAN_AVAILABLE:
            self.godelian = GodelianAntiStagnation(
                stagnation_threshold=0.01,
                window_size=10
            )
            logger.info("✅ Gödelian anti-stagnation initialized")
        else:
            self.godelian = None
        
        # Trajectory tracking for meta-learning
        self.trajectory = []
        
        logger.info(f"✅ System V3.0 initialized at cycle {self.cycle}")
        logger.info(f"📊 Best MNIST: {self.best['mnist']:.1f}% | Best CartPole: {self.best['cartpole']:.1f}")
    
    def run_cycle(self):
        """Execute one complete cycle with all integrations"""
        self.cycle += 1
        logger.info("")
        logger.info("="*80)
        logger.info(f"🔄 CYCLE {self.cycle}")
        logger.info("="*80)
        
        # 1. Train MNIST
        mnist_metrics = self._train_mnist()
        
        # 2. Train CartPole (PPO or DQN)
        cartpole_metrics = self._train_cartpole_advanced()
        
        # 3. Meta-learning (if available)
        if self.meta_learner:
            meta_metrics = self._meta_learn(mnist_metrics, cartpole_metrics)
        else:
            meta_metrics = {}
        
        # 4. Anti-stagnation check (if available)
        if self.godelian:
            stagnation_actions = self._check_stagnation(mnist_metrics, cartpole_metrics)
        else:
            stagnation_actions = []
        
        # 5. Save cycle
        self.db.save_cycle(
            self.cycle,
            mnist=mnist_metrics['test'],
            cartpole=cartpole_metrics['reward'],
            cartpole_avg=cartpole_metrics['avg_reward']
        )
        
        # 6. Check records
        self._check_records(mnist_metrics, cartpole_metrics)
        
        # 7. Save models
        if self.cycle % CHECKPOINT_INTERVAL == 0:
            self._save_all_models()
        
        # 8. Consult APIs (more frequently with LiteLLM)
        api_interval = 10 if LITELLM_AVAILABLE else API_CALL_INTERVAL
        if self.cycle % api_interval == 0:
            self._consult_apis_advanced(mnist_metrics, cartpole_metrics)
        
        # 9. Update trajectory
        self.trajectory.append({
            'cycle': self.cycle,
            'mnist': mnist_metrics['test'],
            'cartpole': cartpole_metrics['avg_reward'],
            'reward': mnist_metrics['test'] + cartpole_metrics['avg_reward']
        })
        
        if len(self.trajectory) > 50:
            self.trajectory = self.trajectory[-50:]
        
        return {
            'mnist': mnist_metrics,
            'cartpole': cartpole_metrics,
            'meta': meta_metrics,
            'stagnation_actions': stagnation_actions
        }
    
    def _train_mnist(self) -> Dict[str, float]:
        """Train MNIST with potential AutoKeras optimization"""
        logger.info("🧠 Training MNIST...")
        
        train_acc = self.mnist.train_epoch()
        test_acc = self.mnist.evaluate()
        
        logger.info(f"   Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
        return {"train": train_acc, "test": test_acc}
    
    def _train_cartpole_advanced(self, episodes: int = 5) -> Dict[str, float]:
        """Train CartPole with PPO (advanced) or DQN (fallback)"""
        if PPO_AVAILABLE:
            return self._train_cartpole_ppo(episodes)
        else:
            return self._train_cartpole_dqn(episodes)
    
    def _train_cartpole_ppo(self, episodes: int = 5) -> Dict[str, float]:
        """Train with CleanRL PPO"""
        logger.info("🎮 Training CartPole with PPO (CleanRL)...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, log_prob, value = self.rl_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rl_agent.store_transition(state, action, reward, float(done), log_prob, value)
                
                total_reward += reward
                state = next_state
            
            # PPO update at end of episode
            if len(self.rl_agent.states) >= self.rl_agent.batch_size:
                loss_info = self.rl_agent.update(state)
            
            episode_rewards.append(total_reward)
            self.cartpole_rewards.append(total_reward)
            self.rl_agent.episode_rewards.append(total_reward)
        
        avg_reward = sum(self.cartpole_rewards) / len(self.cartpole_rewards)
        last_reward = episode_rewards[-1]
        
        logger.info(f"   Last: {last_reward:.1f} | Avg(100): {avg_reward:.1f} | PPO")
        
        return {"reward": last_reward, "avg_reward": avg_reward}
    
    def _train_cartpole_dqn(self, episodes: int = 5) -> Dict[str, float]:
        """Fallback to DQN if PPO not available"""
        logger.info("🎮 Training CartPole with DQN (fallback)...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.rl_agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rl_agent.store_transition(state, action, reward, next_state, float(done))
                loss = self.rl_agent.train_step()
                
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            self.cartpole_rewards.append(total_reward)
        
        avg_reward = sum(self.cartpole_rewards) / len(self.cartpole_rewards)
        last_reward = episode_rewards[-1]
        
        logger.info(f"   Last: {last_reward:.1f} | Avg(100): {avg_reward:.1f} | ε: {self.rl_agent.epsilon:.3f}")
        
        return {"reward": last_reward, "avg_reward": avg_reward}
    
    def _meta_learn(self, mnist_metrics: Dict, cartpole_metrics: Dict) -> Dict[str, Any]:
        """
        Meta-learning cycle
        Agent learns about its own learning
        """
        logger.info("🧠 Meta-learning cycle...")
        
        # Create meta-state from recent performance
        meta_state = np.array([
            mnist_metrics['test'] / 100.0,
            mnist_metrics['train'] / 100.0,
            cartpole_metrics['avg_reward'] / 500.0,
            self.cycle / 1000.0,
            len(self.trajectory) / 50.0,
            self.best['mnist'] / 100.0,
            self.best['cartpole'] / 500.0,
            self.db.get_stagnation_score(10),
            np.random.random(),  # Exploration
            np.random.random()   # Exploration
        ])
        
        # Meta-action selection
        meta_action = self.meta_learner.select_action(meta_state)
        
        # Interpret meta-action
        actions_map = {
            0: 'maintain',
            1: 'increase_learning',
            2: 'decrease_learning',
            3: 'architecture_change',
            4: 'exploration_boost'
        }
        
        action_name = actions_map.get(meta_action, 'maintain')
        
        # Calculate meta-reward (improvement)
        if len(self.trajectory) >= 2:
            prev_perf = self.trajectory[-2]['reward']
            curr_perf = mnist_metrics['test'] + cartpole_metrics['avg_reward']
            meta_reward = (curr_perf - prev_perf) / 100.0  # Normalized
        else:
            meta_reward = 0.0
        
        # Meta-learning update
        next_meta_state = meta_state.copy()
        loss = self.meta_learner.learn(meta_state, meta_action, meta_reward, next_meta_state, False)
        
        # Detect patterns
        pattern = self.meta_learner.detect_pattern(self.trajectory)
        
        # Adapt architecture based on performance
        performance = (mnist_metrics['test'] + cartpole_metrics['avg_reward']) / 600.0
        self.meta_learner.adapt_architecture(performance)
        
        meta_metrics = self.meta_learner.get_meta_metrics()
        
        logger.info(f"   Meta-action: {action_name} | Reward: {meta_reward:.3f} | Patterns: {meta_metrics['patterns_detected']}")
        
        return {
            'action': action_name,
            'reward': meta_reward,
            'metrics': meta_metrics,
            'pattern': pattern
        }
    
    def _check_stagnation(self, mnist_metrics: Dict, cartpole_metrics: Dict) -> List[str]:
        """
        Gödelian anti-stagnation check
        """
        # Combined metric
        combined_metric = (mnist_metrics['test'] + cartpole_metrics['avg_reward']) / 2.0
        
        is_stagnant = self.godelian.is_stagnant(combined_metric)
        
        actions_taken = []
        
        if is_stagnant:
            logger.warning("⚠️  STAGNATION DETECTED - Activating Gödelian protocols")
            
            # Get anti-stagnation action
            action = self.godelian.get_antistagnation_action()
            strategy = action['strategy']
            
            # Apply strategy
            if strategy == 'increase_lr':
                for param_group in self.mnist.optimizer.param_groups:
                    param_group['lr'] *= 1.5
                actions_taken.append('increased_lr')
            
            elif strategy == 'decrease_lr':
                for param_group in self.mnist.optimizer.param_groups:
                    param_group['lr'] *= 0.7
                actions_taken.append('decreased_lr')
            
            elif strategy == 'add_noise':
                # Add noise to parameters
                with torch.no_grad():
                    for param in self.mnist.model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
                actions_taken.append('added_noise')
            
            elif strategy == 'architecture_change':
                if self.meta_learner:
                    # Trigger architecture adaptation
                    self.meta_learner.adapt_architecture(0.5)
                actions_taken.append('architecture_change')
            
            elif strategy == 'reset_optimizer':
                # Reset optimizer state
                self.mnist.optimizer = torch.optim.Adam(
                    self.mnist.model.parameters(),
                    lr=MNIST_CONFIG['lr']
                )
                actions_taken.append('reset_optimizer')
            
            logger.info(f"   🎯 Applied: {actions_taken}")
            
            # Get synergistic actions
            current_metrics = {
                'mnist_test': mnist_metrics['test'],
                'cartpole_avg': cartpole_metrics['avg_reward']
            }
            synergistic = self.godelian.get_synergistic_actions(current_metrics)
            logger.info(f"   💡 Synergistic suggestions: {synergistic}")
        
        return actions_taken
    
    def _check_records(self, mnist_metrics: Dict, cartpole_metrics: Dict):
        """Check and update records"""
        if mnist_metrics['test'] > self.best['mnist']:
            self.best['mnist'] = mnist_metrics['test']
            logger.info(f"   🏆 NEW MNIST RECORD: {mnist_metrics['test']:.2f}%")
        
        if cartpole_metrics['avg_reward'] > self.best['cartpole']:
            self.best['cartpole'] = cartpole_metrics['avg_reward']
            logger.info(f"   🏆 NEW CARTPOLE RECORD: {cartpole_metrics['avg_reward']:.1f}")
    
    def _consult_apis_advanced(self, mnist_metrics: Dict, cartpole_metrics: Dict):
        """
        Advanced API consultation with LiteLLM
        Calls ALL 6 APIs if litellm available
        """
        logger.info("🌐 Consulting APIs (LiteLLM multi-call)...")
        
        recent_cycles = self.db.get_recent_cycles(5)
        metrics = {
            "mnist_test": mnist_metrics['test'],
            "mnist_train": mnist_metrics['train'],
            "cartpole_avg": cartpole_metrics['avg_reward'],
            "cartpole_last": cartpole_metrics['reward'],
            "epsilon": getattr(self.rl_agent, 'epsilon', 0.0),
            "cycle": self.cycle,
            "stagnation": self.db.get_stagnation_score(10),
            "recent_mnist": [c['mnist_accuracy'] for c in recent_cycles if c['mnist_accuracy']],
            "recent_cartpole": [c['cartpole_avg_reward'] for c in recent_cycles if c['cartpole_avg_reward']]
        }
        
        suggestions = self.api_manager.consult_for_improvement(metrics)
        
        self._apply_suggestions_advanced(suggestions)
        
        # Save ALL API responses
        for reasoning in suggestions.get("reasoning", []):
            self.db.save_api_response(
                self.cycle,
                reasoning["api"],
                "improvement_consultation",
                reasoning.get("response", reasoning.get("analysis", "")),
                "parameter_tuning"
            )
        
        api_count = len(suggestions.get("reasoning", []))
        logger.info(f"   ✅ Consulted {api_count} APIs")
    
    def _apply_suggestions_advanced(self, suggestions: Dict[str, Any]):
        """Apply suggestions from all APIs"""
        applied = []
        
        if suggestions.get("increase_lr"):
            for param_group in self.mnist.optimizer.param_groups:
                param_group['lr'] *= 1.3
            applied.append("increased LR")
        
        if suggestions.get("decrease_lr"):
            for param_group in self.mnist.optimizer.param_groups:
                param_group['lr'] *= 0.7
            applied.append("decreased LR")
        
        if suggestions.get("increase_exploration") and hasattr(self.rl_agent, 'epsilon'):
            self.rl_agent.epsilon = min(0.5, self.rl_agent.epsilon * 1.3)
            applied.append("increased exploration")
        
        if suggestions.get("decrease_exploration") and hasattr(self.rl_agent, 'epsilon'):
            self.rl_agent.epsilon = max(0.01, self.rl_agent.epsilon * 0.7)
            applied.append("decreased exploration")
        
        if suggestions.get("architecture_change") and self.meta_learner:
            self.meta_learner.adapt_architecture(0.5)
            applied.append("adapted architecture")
        
        if applied:
            logger.info(f"   ✅ Applied: {', '.join(applied)}")
        else:
            logger.info(f"   ℹ️  No adjustments needed")
    
    def _save_all_models(self):
        """Save all models and checkpoints"""
        logger.info("💾 Saving all models...")
        
        self.mnist.save()
        self.rl_agent.save()
        
        if self.meta_learner:
            self.meta_learner.save()
        
        logger.info("   ✅ All models saved")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'cycle': self.cycle,
            'best_mnist': self.best['mnist'],
            'best_cartpole': self.best['cartpole'],
            'integrations': {
                'ppo': PPO_AVAILABLE,
                'litellm': LITELLM_AVAILABLE,
                'meta_learner': META_LEARNER_AVAILABLE,
                'godelian': GODELIAN_AVAILABLE
            },
            'trajectory_length': len(self.trajectory)
        }
        
        if self.meta_learner:
            status['meta_metrics'] = self.meta_learner.get_meta_metrics()
        
        if self.godelian:
            status['godelian_status'] = self.godelian.get_status()
        
        return status
    
    def run_forever(self):
        """Run system indefinitely"""
        logger.info("🚀 Starting V3.0 continuous operation...")
        
        # Print status
        status = self.get_system_status()
        logger.info(f"📊 System status: {status}")
        
        try:
            while True:
                try:
                    results = self.run_cycle()
                    time.sleep(CYCLE_INTERVAL)
                    
                except KeyboardInterrupt:
                    raise
                    
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}", exc_info=True)
                    self.db.save_error(
                        component="system_v3",
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
        self._save_all_models()
        self.env.close()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    import torch
    
    logger.info("🔥 LAUNCHING INTELLIGENCE SYSTEM V3.0 - MEGA-MERGE")
    logger.info("   Integrating: LiteLLM, CleanRL, Meta-Learner, Gödelian")
    logger.info("")
    
    system = IntelligenceSystemV3()
    system.run_forever()
