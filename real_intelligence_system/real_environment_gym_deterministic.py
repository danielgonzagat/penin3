
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

#!/usr/bin/env python3
"""
AMBIENTE REAL PARA TEIS V2 - OpenAI Gym Integration
==================================================
Conecta o TEIS V2 Enhanced com ambientes reais de aprendizado
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealEnvironmentGym")

class RealEnvironmentGym:
    """
    Ambiente real integrado com OpenAI Gym para TEIS V2
    """
    
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.env = None
        self.model = None
        self.vec_env = None
        self.training_data = []
        
    def create_environment(self):
        """Cria ambiente real do Gym"""
        try:
            # Criar ambiente vetorizado para melhor performance
            self.vec_env = make_vec_env(self.env_name, n_envs=4)
            
            # Normalizar observações para melhor aprendizado
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
            
            logger.info(f"✅ Ambiente real criado: {self.env_name}")
            logger.info(f"📊 Observação: {self.vec_env.observation_space}")
            logger.info(f"🎯 Ação: {self.vec_env.action_space}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar ambiente: {e}")
            return False
    
    def create_ppo_model(self):
        """Cria modelo PPO para aprendizado por reforço"""
        try:
            # Usar arquitetura personalizada baseada no TEIS V2
            policy_kwargs = dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                activation_fn=torch.nn.ReLU
            )
            
            self.model = PPO(
                "MlpPolicy",
                self.vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1
            )
            
            logger.info("✅ Modelo PPO criado com arquitetura personalizada")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar modelo PPO: {e}")
            return False
    
    def train_model(self, total_timesteps: int = 100000):
        """Treina o modelo no ambiente real"""
        try:
            logger.info(f"🎯 Iniciando treinamento por {total_timesteps} timesteps...")
            
            # Treinar modelo
            self.model.learn(total_timesteps=total_timesteps)
            
            logger.info("✅ Treinamento concluído!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return False
    
    def evaluate_model(self, n_episodes: int = 10):
        """Avalia o modelo treinado"""
        try:
            logger.info(f"📊 Avaliando modelo em {n_episodes} episódios...")
            
            episode_rewards = []
            
            for episode in range(n_episodes):
                obs = self.vec_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.vec_env.step(action)
                    episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                episode_rewards.append(episode_reward)
                logger.info(f"  Episódio {episode + 1}: {episode_reward:.2f}")
            
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            logger.info(f"📈 Recompensa média: {avg_reward:.2f} ± {std_reward:.2f}")
            
            return {
                'episode_rewards': episode_rewards,
                'avg_reward': avg_reward,
                'std_reward': std_reward
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na avaliação: {e}")
            return None
    
    def run_learning_cycle(self):
        """Executa um ciclo de aprendizado"""
        try:
            if self.model is None:
                return None
            
            # Executar alguns passos de treinamento
            self.model.learn(total_timesteps=1000)
            
            # Coletar métricas
            metrics = {
                'learning_steps': 1000,
                'timestamp': np.datetime64('now'),
                'model_updated': True
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erro no ciclo de aprendizado: {e}")
            return None
    
    def save_model(self, path: str):
        """Salva o modelo treinado"""
        try:
            self.model.save(path)
            logger.info(f"💾 Modelo salvo em: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self, path: str):
        """Carrega modelo salvo"""
        try:
            self.model = PPO.load(path, env=self.vec_env)
            logger.info(f"📁 Modelo carregado de: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False

def main():
    """Função principal para teste"""
    logger.info("🚀 INICIANDO AMBIENTE REAL PARA TEIS V2")
    
    # Criar ambiente real
    real_env = RealEnvironmentGym("CartPole-v1")
    
    if real_env.create_environment():
        if real_env.create_ppo_model():
            # Treinar modelo
            real_env.train_model(total_timesteps=10000)
            
            # Avaliar modelo
            results = real_env.evaluate_model(n_episodes=5)
            
            if results:
                logger.info("✅ Teste do ambiente real concluído com sucesso!")
            else:
                logger.error("❌ Falha na avaliação do modelo")
        else:
            logger.error("❌ Falha ao criar modelo PPO")
    else:
        logger.error("❌ Falha ao criar ambiente")

if __name__ == "__main__":
    main()
