#!/usr/bin/env python3
"""
Multi-Environment Daemon
Alterna entre CartPole e LunarLander para testar generalização
"""
import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')

from brain_daemon_real_env import RealEnvironmentBrainV3
import time

class MultiEnvBrain:
    def __init__(self):
        self.envs = ['CartPole-v1', 'LunarLander-v2']
        self.current_env_idx = 0
        self.brain = None
        
    def switch_env(self):
        """Troca de ambiente a cada 20 episódios"""
        self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)
        env_name = self.envs[self.current_env_idx]
        print(f"\n{'='*60}")
        print(f"🔄 SWITCHING TO: {env_name}")
        print(f"{'='*60}\n")
        
        # Criar novo brain com ambiente diferente
        self.brain = RealEnvironmentBrainV3(env_name=env_name, learning_rate=3e-4)
        self.brain.initialize()
        
    def run(self, episodes_per_env=20):
        """Roda alternando ambientes"""
        total_episodes = 0
        
        while True:
            # Iniciar com ambiente atual
            self.switch_env()
            
            # Rodar N episódios neste ambiente
            for _ in range(episodes_per_env):
                if not self.brain.running:
                    break
                try:
                    self.brain.run_episode()
                    total_episodes += 1
                except KeyboardInterrupt:
                    print("\n⏸️  Interrompido pelo usuário")
                    return
                except Exception as e:
                    print(f"❌ Erro: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            if not self.brain.running:
                break

if __name__ == '__main__':
    multi_brain = MultiEnvBrain()
    multi_brain.run(episodes_per_env=20)