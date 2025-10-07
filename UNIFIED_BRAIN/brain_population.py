#!/usr/bin/env python3
"""
🧬 POPULATION-BASED BRAIN
Múltiplos cérebros competindo → Evolução real
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')

import torch
import copy
import random
from pathlib import Path
import json

try:
    import gymnasium as gym
except:
    import gym

from unified_brain_core import CoreSoupHybrid
from brain_system_integration import UnifiedSystemController
from brain_logger import brain_logger

class PopulationBrain:
    """
    População de cérebros competindo
    Seleção natural + mutação = evolução REAL
    """
    
    def __init__(self, population_size=10, env_name='CartPole-v1'):
        self.population_size = population_size
        self.env_name = env_name
        self.generation = 0
        self.population = []
        
        brain_logger.info(f"Initializing population: {population_size} brains")
        
        # Cria população inicial
        for i in range(population_size):
            agent = self.create_agent(agent_id=i, generation=0)
            self.population.append(agent)
        
        brain_logger.info("Population initialized!")
    
    def create_agent(self, agent_id, generation):
        """Cria um agente (cérebro + controller)"""
        hybrid = CoreSoupHybrid(H=1024)
        
        # Carrega base
        snapshot = Path("/root/UNIFIED_BRAIN/snapshots/initial_state_registry.json")
        if snapshot.exists():
            hybrid.core.registry.load_with_adapters(str(snapshot))
            hybrid.core.initialize_router()
        
        controller = UnifiedSystemController(hybrid.core)
        controller.connect_v7(obs_dim=4, act_dim=2)
        
        return {
            'id': agent_id,
            'generation': generation,
            'brain': hybrid,
            'controller': controller,
            'fitness': 0.0,
            'age': 0,
            'wins': 0,
            'genome': self.extract_genome(hybrid)
        }
    
    def extract_genome(self, hybrid):
        """Extrai 'DNA' do cérebro (para tracking)"""
        # Usa competence scores como genome
        if hybrid.core.router:
            return hybrid.core.router.competence.clone().detach()
        return torch.zeros(10)
    
    def evaluate_agent(self, agent, env, episodes=3):
        """
        Avalia fitness de um agente
        Fitness = média de reward em N episódios
        """
        total_reward = 0.0
        
        for ep in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                obs = torch.FloatTensor(state).unsqueeze(0)
                
                result = agent['controller'].step(
                    obs=obs,
                    reward=episode_reward / 500.0
                )
                
                action = result['action_logits'].argmax(-1).item()
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_result
                
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
        
        fitness = total_reward / episodes
        return fitness
    
    def run_generation(self):
        """
        Roda uma geração completa:
        1. Avalia todos
        2. Seleção natural
        3. Reprodução com mutação
        """
        brain_logger.info(f"="*80)
        brain_logger.info(f"Generation {self.generation}")
        brain_logger.info(f"="*80)
        
        # 1. AVALIAÇÃO
        env = gym.make(self.env_name)
        
        for i, agent in enumerate(self.population):
            fitness = self.evaluate_agent(agent, env, episodes=3)
            agent['fitness'] = fitness
            agent['age'] += 1
            
            if (i + 1) % 3 == 0:
                brain_logger.info(f"   Evaluated {i+1}/{self.population_size} agents...")
        
        env.close()
        
        # 2. ORDENAÇÃO
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        best = self.population[0]
        worst = self.population[-1]
        avg = sum(a['fitness'] for a in self.population) / len(self.population)
        
        brain_logger.info(f"")
        brain_logger.info(f"Results:")
        brain_logger.info(f"   Best:  {best['fitness']:.2f} (id={best['id']})")
        brain_logger.info(f"   Avg:   {avg:.2f}")
        brain_logger.info(f"   Worst: {worst['fitness']:.2f} (id={worst['id']})")
        
        # 3. SELEÇÃO NATURAL (top 50% sobrevive)
        survivors = self.population[:self.population_size // 2]
        brain_logger.info(f"   Survivors: {len(survivors)}/{self.population_size}")
        
        # 4. REPRODUÇÃO (crossover + mutação)
        children = []
        for i in range(self.population_size // 2):
            # Parent aleatório dos survivors
            parent = random.choice(survivors)
            
            # Clone + mutação
            child = self.reproduce(parent, child_id=self.population_size + i)
            children.append(child)
        
        # 5. NOVA POPULAÇÃO
        self.population = survivors + children
        
        # 6. Incrementa geração
        self.generation += 1
        
        # 7. Salva checkpoint
        self.save_generation()
        
        brain_logger.info(f"Generation {self.generation} complete!")
        brain_logger.info(f"")
        
        return best['fitness']
    
    def reproduce(self, parent, child_id):
        """
        Reprodução: clone + mutação
        """
        # Clone profundo
        child_hybrid = copy.deepcopy(parent['brain'])
        
        # MUTAÇÃO nos adapters
        mutation_rate = 0.05
        
        for neuron in child_hybrid.core.registry.get_active()[:50]:  # Primeiros 50
            if random.random() < mutation_rate:
                # Mutação leve
                for param in neuron.A_in.parameters():
                    noise = torch.randn_like(param) * 0.02
                    param.data += noise
                
                for param in neuron.A_out.parameters():
                    noise = torch.randn_like(param) * 0.02
                    param.data += noise
        
        # MUTAÇÃO no router
        if child_hybrid.core.router:
            noise = torch.randn_like(child_hybrid.core.router.competence) * 0.1
            child_hybrid.core.router.competence.data += noise
            child_hybrid.core.router.competence.clamp_(0.0, 10.0)
        
        # Cria controller
        controller = UnifiedSystemController(child_hybrid.core)
        controller.connect_v7(obs_dim=4, act_dim=2)
        
        return {
            'id': child_id,
            'generation': self.generation + 1,
            'brain': child_hybrid,
            'controller': controller,
            'fitness': 0.0,
            'age': 0,
            'wins': 0,
            'genome': self.extract_genome(child_hybrid),
            'parent_id': parent['id']
        }
    
    def save_generation(self):
        """Salva estado da geração"""
        gen_data = {
            'generation': self.generation,
            'population_size': len(self.population),
            'fitnesses': [a['fitness'] for a in self.population],
            'best_fitness': self.population[0]['fitness'],
            'avg_fitness': sum(a['fitness'] for a in self.population) / len(self.population)
        }
        
        gen_path = Path(f"/root/UNIFIED_BRAIN/population_gen_{self.generation}.json")
        with open(gen_path, 'w') as f:
            json.dump(gen_data, f, indent=2)


if __name__ == "__main__":
    print("Testing Population-Based Training...")
    
    pop = PopulationBrain(population_size=5)
    
    # Roda 3 gerações
    for gen in range(3):
        best_fitness = pop.run_generation()
        print(f"Generation {gen}: best_fitness={best_fitness:.2f}")
    
    print("✅ Population training OK!")
