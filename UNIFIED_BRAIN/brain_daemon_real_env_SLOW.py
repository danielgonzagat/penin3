#!/usr/bin/env python3
"""
🔥 BRAIN DAEMON COM AMBIENTE REAL - VERSÃO COM APRENDIZADO REAL
CORRIGIDO: Gradientes + Curiosity + Performance + Logging
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import torch
import torch.nn as nn
import time
from pathlib import Path
import signal
import json
from datetime import datetime
from collections import deque

try:
    import gymnasium as gym
except:
    import gym

from unified_brain_core import CoreSoupHybrid
from brain_system_integration import UnifiedSystemController
from brain_logger import brain_logger
from curiosity_module import CuriosityModule
from intelligence_system.config.settings import (
    MULTITASK_ENVS,
    MULTITASK_ROTATION_EPISODES,
    SHADOW_EPISODES,
    KILL_SWITCH_PATH,
)

class RealEnvironmentBrainV2:
    """
    Daemon com AMBIENTE REAL + APRENDIZADO REAL + CURIOSITY
    """
    
    def __init__(self, env_name='CartPole-v1', learning_rate=1e-4, use_gpu=True):
        self.running = True
        self.hybrid = None
        self.controller = None
        self.curiosity = None
        self.optimizer = None
        
        # 🔥 GPU SUPPORT
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            brain_logger.info("🚀 Using GPU (CUDA)")
        else:
            self.device = torch.device('cpu')
            brain_logger.info("💻 Using CPU")
        
        # AMBIENTE REAL (com rotação mínima multi-tarefa)
        self.env_names = list(dict.fromkeys([env_name] + MULTITASK_ENVS))
        self._env_index = 0
        try:
            self.env = gym.make(self.env_names[self._env_index])
        except Exception:
            import gym as old_gym
            self.env = old_gym.make(self.env_names[self._env_index])
        
        self.state = None
        self.episode_reward = 0
        self.episode = 0
        self.best_reward = 0
        self.learning_rate = learning_rate
        
        # Stats
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_episodes': 0,
            'rewards': [],
            'best_reward': 0,
            'avg_reward_last_100': 0,
            'learning_progress': 0.0,
            'total_curiosity': 0.0,
            'gradients_applied': 0,
            'avg_loss': 0.0,
            'device': str(self.device)
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        brain_logger.info(f"🔥 Real Environment Brain V2: {env_name}")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        brain_logger.info(f"Shutting down. Episodes: {self.episode}, Best: {self.best_reward:.1f}")
        self.save_checkpoint()
        self.running = False
    
    def initialize(self):
        """Inicializa cérebro COM APRENDIZADO"""
        brain_logger.info("Loading brain...")
        
        self.hybrid = CoreSoupHybrid(H=1024)
        
        snapshot_path = Path("/root/UNIFIED_BRAIN/snapshots/initial_state_registry.json")
        if snapshot_path.exists():
            self.hybrid.core.registry.load_with_adapters(str(snapshot_path))
            self.hybrid.core.initialize_router()
            brain_logger.info(f"Brain loaded: {self.hybrid.core.registry.count()['total']} neurons")
        else:
            brain_logger.warning("No snapshot, starting fresh")
        
        self.controller = UnifiedSystemController(self.hybrid.core)
        self.controller.connect_v7(obs_dim=4, act_dim=2)
        
        # 🔥 CURIOSITY MODULE
        self.curiosity = CuriosityModule(H=1024, action_dim=2)
        
        # 🔥 Move para GPU (se disponível)
        if self.device.type == 'cuda':
            self.curiosity = self.curiosity.to(self.device)
            brain_logger.info("Models moved to GPU")
        brain_logger.info("✅ Curiosity module initialized")
        
        # 🔥 OPTIMIZER para APRENDIZADO
        trainable_params = []
        
        # Adapters dos top neurons
        for neuron in self.hybrid.core.registry.get_active()[:64]:
            trainable_params.extend(list(neuron.A_in.parameters()))
            trainable_params.extend(list(neuron.A_out.parameters()))
        
        # V7 bridge
        trainable_params.extend(list(self.controller.v7_bridge.parameters()))
        
        # Router competence
        if self.hybrid.core.router:
            trainable_params.append(self.hybrid.core.router.competence)
        
        # Curiosity
        trainable_params.extend(list(self.curiosity.parameters()))
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        brain_logger.info(f"✅ Optimizer initialized: {len(trainable_params)} parameters")
        
        # Reset environment
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        
        brain_logger.info("✅ Real environment ready with LEARNING!")
    
    def run_episode(self):
        """
        Roda UM episódio com APRENDIZADO REAL + CURIOSITY
        """
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        
        episode_reward = 0
        episode_curiosity = 0
        steps = 0
        done = False
        
        # Buffers do episódio
        ep_states, ep_actions, ep_rewards = [], [], []
        ep_values, ep_log_probs = [], []
        
        while not done and self.running and steps < 500:
            # Kill-switch check
            try:
                from pathlib import Path as _Path
                if _Path(KILL_SWITCH_PATH).exists():
                    brain_logger.critical("KILL SWITCH detected; stopping episode")
                    self.running = False
                    break
            except Exception:
                pass
            # 1. Estado REAL
            obs = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
            
            # 2. Forward no cérebro
            result = self.controller.step(
                obs=obs,
                penin_metrics={
                    'L_infinity': episode_reward / 500.0,
                    'CAOS_plus': 1.0 - (episode_reward / 500.0),
                    'SR_Omega_infinity': 0.7
                },
                reward=episode_reward / 500.0
            )
            
            action_logits = result['action_logits']
            value = result['value']
            Z = result['Z']
            
            # 3. Sample action (com exploration)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            action_int = action.item()
            
            # 4. 🔥 CURIOSITY REWARD
            curiosity_reward = self.curiosity.compute_curiosity(Z, action_int)
            episode_curiosity += curiosity_reward
            
            # 5. Environment step
            step_result = self.env.step(action_int)
            
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            # 6. Total reward (env + curiosity)
            total_reward = reward + curiosity_reward * 0.1
            
            # 7. Store experience
            ep_states.append(obs)
            ep_actions.append(action)
            ep_rewards.append(total_reward)
            ep_values.append(value)
            ep_log_probs.append(log_prob)
            
            # 8. Update
            self.state = next_state
            episode_reward += reward
            steps += 1
            self.stats['total_steps'] += 1
        
        # 🔥 APRENDIZADO: Calcula gradientes e atualiza
        loss_value = 0.0
        if len(ep_rewards) > 1:
            loss_value = self.train_on_episode(ep_states, ep_actions, ep_rewards, ep_values, ep_log_probs)
        
        # Stats
        self.episode += 1
        self.stats['total_episodes'] += 1
        self.stats['rewards'].append(episode_reward)
        self.stats['total_curiosity'] += episode_curiosity
        
        if len(self.stats['rewards']) > 1000:
            self.stats['rewards'] = self.stats['rewards'][-1000:]
        
        # Best
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.stats['best_reward'] = self.best_reward
            brain_logger.info(f"🎊 NEW BEST: {self.best_reward:.1f}!")
        
        # Avg
        recent = self.stats['rewards'][-100:]
        self.stats['avg_reward_last_100'] = sum(recent) / len(recent)
        
        # Progress
        threshold = 195.0
        if len(recent) >= 10:
            solved = sum(1 for r in recent if r >= threshold)
            self.stats['learning_progress'] = solved / len(recent)
        
        # 🔥 LOG CADA EPISÓDIO
        brain_logger.info(
            f"Episode {self.episode}: "
            f"reward={episode_reward:.1f}, "
            f"curiosity={episode_curiosity:.2f}, "
            f"loss={loss_value:.4f}, "
            f"avg_100={self.stats['avg_reward_last_100']:.1f}, "
            f"best={self.best_reward:.1f}, "
            f"steps={steps}"
        )
        
        # 🔥 CHECKPOINT a cada 10 episódios
        if self.episode % 10 == 0:
            self.save_checkpoint()
        
        # 🔁 Rotação de ambiente mínima para generalização
        try:
            if self.episode % max(1, MULTITASK_ROTATION_EPISODES) == 0:
                self._env_index = (self._env_index + 1) % max(1, len(self.env_names))
                next_env = self.env_names[self._env_index]
                try:
                    self.env.close()
                except Exception:
                    pass
                try:
                    self.env = gym.make(next_env)
                    brain_logger.info(f"🔁 Switched environment → {next_env}")
                except Exception as e:
                    brain_logger.warning(f"Failed to switch env to {next_env}: {e}")
        except Exception:
            pass
        
        return episode_reward
    
    def train_on_episode(self, states, actions, rewards, values, log_probs):
        """
        🔥 APRENDIZADO REAL: Loss + Backward + Optimizer
        """
        # Returns (discounted)
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normaliza
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Concatena
        states_batch = torch.cat(states, dim=0)
        actions_batch = torch.stack(actions)
        values_batch = torch.cat([v for v in values], dim=0).squeeze()
        log_probs_batch = torch.stack(log_probs)
        
        # Advantages
        advantages = returns - values_batch.detach()
        
        # Losses
        policy_loss = -(log_probs_batch * advantages).mean()
        value_loss = ((values_batch - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # 🔥 BACKWARD + OPTIMIZER STEP
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
            max_norm=0.5
        )
        
        self.optimizer.step()
        
        self.stats['gradients_applied'] += 1
        self.stats['avg_loss'] = 0.9 * self.stats['avg_loss'] + 0.1 * loss.item()
        
        return loss.item()
    
    def save_checkpoint(self):
        """Salva checkpoint"""
        checkpoint = {
            'stats': self.stats,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = Path("/root/UNIFIED_BRAIN/real_env_checkpoint.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Salva pesos
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'episode': self.episode
        }, "/root/UNIFIED_BRAIN/real_env_weights.pt")
        
        brain_logger.info(f"💾 Checkpoint saved: episode {self.episode}")
    
    def run(self):
        """Loop principal - APRENDIZADO REAL"""
        self.initialize()
        
        brain_logger.info("="*80)
        brain_logger.info("🔥 REAL LEARNING STARTING (WITH GRADIENTS!)")
        brain_logger.info("="*80)
        brain_logger.info(f"Device: {self.device}")
        brain_logger.info("Environment: CartPole-v1")
        brain_logger.info("Goal: Learn to balance (reward > 195)")
        brain_logger.info("Features: Gradients ✅, Curiosity ✅, GPU ✅")
        brain_logger.info("Press Ctrl+C to stop")
        brain_logger.info("="*80)
        
        while self.running:
            try:
                episode_reward = self.run_episode()
                
            except Exception as e:
                brain_logger.error(f"Error in episode: {e}")
                import traceback
                traceback.print_exc()
                self.state = self.env.reset()
                if isinstance(self.state, tuple):
                    self.state = self.state[0]
        
        # Cleanup
        self.save_checkpoint()
        brain_logger.info(f"Training stopped. Total episodes: {self.episode}")
        brain_logger.info(f"Best reward: {self.best_reward:.1f}")
        brain_logger.info(f"Gradients applied: {self.stats['gradients_applied']}")


if __name__ == "__main__":
    daemon = RealEnvironmentBrainV2()
    daemon.run()
