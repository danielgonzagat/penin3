#!/usr/bin/env python3
"""
🔥 BRAIN DAEMON COM AMBIENTE REAL
Feedback loop real → Emergência de inteligência
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import torch
import time
from pathlib import Path
import signal
import json
from datetime import datetime

try:
    import gymnasium as gym
except:
    import gym

from unified_brain_core import CoreSoupHybrid
from brain_system_integration import UnifiedSystemController
from brain_logger import brain_logger

class RealEnvironmentBrain:
    """
    Daemon com AMBIENTE REAL (CartPole)
    Agora o cérebro PRECISA aprender para ter reward
    """
    
    def __init__(self, env_name='CartPole-v1'):
        self.running = True
        self.hybrid = None
        self.controller = None
        
        # AMBIENTE REAL
        try:
            self.env = gym.make(env_name)
        except:
            # Fallback para versão antiga
            import gym as old_gym
            self.env = old_gym.make(env_name)
        
        self.state = None
        self.episode_reward = 0
        self.episode = 0
        self.best_reward = 0
        
        # Stats
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_episodes': 0,
            'rewards': [],  # Histórico de rewards
            'best_reward': 0,
            'avg_reward_last_100': 0,
            'learning_progress': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        brain_logger.info(f"Real Environment Brain: {env_name}")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        brain_logger.info(f"Shutting down. Episodes: {self.episode}, Best: {self.best_reward:.1f}")
        self.save_checkpoint()
        self.running = False
    
    def initialize(self):
        """Inicializa cérebro"""
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
        
        # Reset environment
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]  # Gymnasium retorna (obs, info)
        
        brain_logger.info("Real environment ready!")
    
    def run_episode(self):
        """
        Roda UM episódio completo no ambiente real
        AQUI O CÉREBRO APRENDE DE VERDADE
        """
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and self.running:
            # 1. Estado REAL do ambiente
            obs = torch.FloatTensor(self.state).unsqueeze(0)
            
            # 2. Cérebro decide ação
            result = self.controller.step(
                obs=obs,
                penin_metrics={
                    'L_infinity': episode_reward / 500.0,
                    'CAOS_plus': 1.0 - (episode_reward / 500.0),  # Explora se reward baixo
                    'SR_Omega_infinity': 0.7
                },
                reward=episode_reward / 500.0  # Normalizado
            )
            
            # 3. Ação no ambiente REAL
            action = result['action_logits'].argmax(-1).item()
            
            # 4. CONSEQUÊNCIA REAL
            step_result = self.env.step(action)
            
            # Compatibilidade Gym/Gymnasium
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            # 5. FEEDBACK LOOP REAL
            self.state = next_state
            episode_reward += reward
            steps += 1
            
            self.stats['total_steps'] += 1
        
        # Fim do episódio
        self.episode += 1
        self.stats['total_episodes'] += 1
        self.stats['rewards'].append(episode_reward)
        
        # Mantém últimos 1000 rewards
        if len(self.stats['rewards']) > 1000:
            self.stats['rewards'] = self.stats['rewards'][-1000:]
        
        # Atualiza best
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.stats['best_reward'] = self.best_reward
            brain_logger.info(f"🎊 NEW BEST: {self.best_reward:.1f}!")
        
        # Média últimos 100
        recent = self.stats['rewards'][-100:]
        self.stats['avg_reward_last_100'] = sum(recent) / len(recent)
        
        # Learning progress (% de episódios acima de threshold)
        threshold = 195.0  # CartPole é resolvido em 195
        if len(recent) >= 10:
            solved = sum(1 for r in recent if r >= threshold)
            self.stats['learning_progress'] = solved / len(recent)
        
        # Log periódico
        if self.episode % 10 == 0:
            brain_logger.info(
                f"Episode {self.episode}: "
                f"reward={episode_reward:.1f}, "
                f"avg_100={self.stats['avg_reward_last_100']:.1f}, "
                f"best={self.best_reward:.1f}, "
                f"progress={self.stats['learning_progress']*100:.1f}%"
            )
        
        # Checkpoint a cada 100 episódios
        if self.episode % 100 == 0:
            self.save_checkpoint()
        
        return episode_reward
    
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
        
        brain_logger.info(f"Checkpoint saved: episode {self.episode}")
    
    def run(self):
        """Loop principal - APRENDIZADO REAL"""
        self.initialize()
        
        brain_logger.info("="*80)
        brain_logger.info("🔥 REAL ENVIRONMENT LEARNING STARTING")
        brain_logger.info("="*80)
        brain_logger.info("Environment: CartPole-v1")
        brain_logger.info("Goal: Learn to balance (reward > 195)")
        brain_logger.info("Press Ctrl+C to stop")
        brain_logger.info("="*80)
        
        while self.running:
            try:
                episode_reward = self.run_episode()
                
            except Exception as e:
                brain_logger.error(f"Error in episode: {e}")
                self.state = self.env.reset()
                if isinstance(self.state, tuple):
                    self.state = self.state[0]
        
        # Cleanup
        self.save_checkpoint()
        brain_logger.info(f"Training stopped. Total episodes: {self.episode}")
        brain_logger.info(f"Best reward: {self.best_reward:.1f}")


if __name__ == "__main__":
    daemon = RealEnvironmentBrain()
    daemon.run()
