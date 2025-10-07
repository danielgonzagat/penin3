"""
Tarefas expandidas para evitar estagnação no CartPole
"""
import gym
import numpy as np
from typing import Dict, Any, Optional

class ExpandedTasks:
    """Gerador de tarefas além do CartPole padrão"""
    
    def __init__(self):
        self.tasks = {
            'cartpole': self._cartpole_standard,
            'cartpole_harder': self._cartpole_harder,
            'cartpole_perturbed': self._cartpole_perturbed,
            'acrobot': self._acrobot,
            'mountain_car': self._mountain_car,
        }
        self.current_task = 'cartpole'
    
    def _cartpole_standard(self) -> gym.Env:
        """CartPole padrão"""
        return gym.make('CartPole-v1')
    
    def _cartpole_harder(self) -> gym.Env:
        """CartPole com física modificada (mais difícil)"""
        env = gym.make('CartPole-v1')
        # Aumentar gravidade ou reduzir limite de ângulo
        if hasattr(env, 'theta_threshold_radians'):
            env.theta_threshold_radians *= 0.8  # Mais difícil
        return env
    
    def _cartpole_perturbed(self) -> gym.Env:
        """CartPole com perturbações aleatórias"""
        env = gym.make('CartPole-v1')
        env.original_step = env.step
        
        def perturbed_step(action):
            # Adicionar ruído à ação
            if np.random.random() < 0.1:
                action = 1 - action
            return env.original_step(action)
        
        env.step = perturbed_step
        return env
    
    def _acrobot(self) -> gym.Env:
        """Acrobot: tarefa de controle mais complexa"""
        try:
            return gym.make('Acrobot-v1')
        except:
            return self._cartpole_standard()
    
    def _mountain_car(self) -> gym.Env:
        """MountainCar: requer estratégia diferente"""
        try:
            return gym.make('MountainCar-v0')
        except:
            return self._cartpole_standard()
    
    def get_next_task(self, current_performance: float) -> str:
        """
        Seleciona próxima tarefa baseado em performance
        
        Se CartPole está dominado (>480), tenta variações
        """
        if current_performance > 480 and self.current_task == 'cartpole':
            # Graduar dificuldade
            options = ['cartpole_harder', 'cartpole_perturbed', 'acrobot', 'mountain_car']
            self.current_task = np.random.choice(options)
        elif current_performance < 300:
            # Voltar ao básico
            self.current_task = 'cartpole'
        
        return self.current_task
    
    def create_env(self, task_name: Optional[str] = None) -> gym.Env:
        """Cria ambiente da tarefa"""
        if task_name is None:
            task_name = self.current_task
        
        return self.tasks.get(task_name, self._cartpole_standard)()
