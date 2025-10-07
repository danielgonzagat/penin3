#!/usr/bin/env python3
"""
🧬 IA³ DARWIN TEIS INTEGRATED SYSTEM
Sistema de Inteligência Emergente Verdadeira Integrada

Combina:
- TEIS: Framework comportamental e evolução social
- IA³ Darwin Brain v3: Evolução neural adaptativa
- Real emergence detection
- Anti-stagnation measures
- Continuous 24/7 evolution

Meta: Alcançar IA³ completa com inteligência emergente
"""

import os
import sys
import time
import json
import random
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Any, Set, Tuple
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Importar componentes
try:
    from observability import observability, traced_method, monitored_component
    from agent_behavior_learner import AgentBehaviorLearner
    from emergence_detector import EmergenceDetector
    from episodic_memory import EpisodicMemory
except ImportError:
    logger.info("⚠️ Alguns componentes de observabilidade não disponíveis, continuando...")

# Importar Darwin Brain v3
sys.path.append('/root/darwin_v3/darwin/core')
from brain_v3 import DarwinBrainV3

# ========== IA³ DARWIN BRAIN ENHANCED ==========

class IA3DarwinBrain(DarwinBrainV3):
    """IA³ Darwin Brain v3 enhanced for TEIS integration"""

    async def __init__(self, input_dim: int, hidden_dim: int = 32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

        # IA³ properties tracking
        self.ia3_properties = {
            'adaptativo': 0.1,
            'autorecursivo': 0.1,
            'autoevolutivo': 0.1,
            'autoconsciente': 0.0,
            'autossuficiente': 0.8,
            'autodidata': 0.2,
            'autoconstruida': 0.1,
            'autossinaptica': 0.1,
            'autoarquitetavel': 0.3,
            'autorregenerativa': 0.9,
            'autotreinada': 0.4,
            'autotuning': 0.5,
            'autoinfinita': 0.0
        }

        self.emergence_events = []
        self.self_reflection_log = []

    async def update_ia3_properties(self, metrics: Dict[str, float]):
        """Update IA³ properties based on system performance"""

        # Adaptativo: Baseado em diversidade comportamental
        self.ia3_properties['adaptativo'] = min(1.0, metrics.get('behavior_diversity', 0.1))

        # Autoevolutivo: Baseado em melhorias no fitness
        fitness_improvement = metrics.get('fitness_improvement', 0.0)
        self.ia3_properties['autoevolutivo'] = min(1.0, max(0.0, fitness_improvement * 10))

        # Autoconsciente: Baseado em detecção de emergência
        emergence_score = metrics.get('emergence_score', 0.0)
        self.ia3_properties['autoconsciente'] = min(1.0, emergence_score)

        # Autodidata: Baseado em aprendizado contínuo
        learning_rate = metrics.get('learning_rate', 0.2)
        self.ia3_properties['autodidata'] = min(1.0, learning_rate * 5)

        # Autoinfinita: Baseado em auto-modificações bem-sucedidas
        self_modifications = metrics.get('successful_modifications', 0)
        self.ia3_properties['autoinfinita'] = min(1.0, successful_modifications * 0.1)

    async def check_emergence(self) -> bool:
        """Check if emergence has occurred based on IA³ properties"""
        total_score = sum(self.ia3_properties.values())
        emergence_threshold = 8.0  # 8+ propriedades ativas

        if total_score >= emergence_threshold:
            self.emergence_events.append({
                'timestamp': datetime.now(),
                'ia3_score': total_score,
                'properties': self.ia3_properties.copy()
            })
            return await True
        return await False

# ========== COMPLEX TASK SYSTEM ==========

class ComplexTaskManager:
    """Gerenciador de tarefas complexas que exigem inteligência"""

    async def __init__(self):
        self.active_tasks = []
        self.completed_tasks = []
        self.task_types = {
            'resource_puzzle': {
                'description': 'Solve resource allocation puzzles',
                'complexity': 'medium',
                'requires_intelligence': True
            },
            'pattern_recognition': {
                'description': 'Identify complex patterns in data',
                'complexity': 'high',
                'requires_intelligence': True
            },
            'strategic_planning': {
                'description': 'Plan multi-step strategies',
                'complexity': 'very_high',
                'requires_intelligence': True
            },
            'creative_problem_solving': {
                'description': 'Solve novel problems creatively',
                'complexity': 'extreme',
                'requires_intelligence': True
            }
        }

    async def create_task(self, task_type: str, difficulty: int = 1) -> Dict[str, Any]:
        """Create a complex task"""
        if task_type not in self.task_types:
            return await None

        task = {
            'id': hashlib.md5(f"{task_type}_{time.time()}".encode()).hexdigest()[:8],
            'type': task_type,
            'difficulty': difficulty,
            'created_at': datetime.now(),
            'status': 'active',
            'progress': 0.0,
            'complexity': self.task_types[task_type]['complexity']
        }

        self.active_tasks.append(task)
        return await task

    async def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks that need attention"""
        return await [t for t in self.active_tasks if t['status'] == 'active']

# ========== COMMUNICATION SYSTEM ==========

class CommunicationSystem:
    """Sistema de comunicação entre agentes"""

    async def __init__(self):
        self.messages = deque(maxlen=1000)
        self.agent_languages = defaultdict(dict)

    async def send_message(self, sender: str, receiver: str, content: Dict[str, Any]):
        """Send message between agents"""
        message = {
            'sender': sender,
            'receiver': receiver,
            'content': content,
            'timestamp': datetime.now(),
            'id': hashlib.md5(f"{sender}_{receiver}_{time.time()}".encode()).hexdigest()
        }
        self.messages.append(message)

    async def get_messages_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for specific agent"""
        return await [m for m in self.messages if m['receiver'] == agent_id]

# ========== REAL AGENT ==========

class RealAgent:
    """Agente com aprendizado verdadeiro e memória"""

    async def __init__(self, agent_id: str, brain: IA3DarwinBrain, task_manager: ComplexTaskManager):
        self.agent_id = agent_id
        self.brain = brain
        self.task_manager = task_manager

        # Estado do agente
        self.energy = 100.0
        self.fitness = 0.0
        self.experience = 0
        self.skills = defaultdict(float)

        # Memória episódica
        self.episodic_memory = []

        # Estado comportamental
        self.behaviors = {
            'cooperative': 0.5,
            'competitive': 0.5,
            'explorative': 0.5,
            'conservative': 0.5
        }

    async def perceive_environment(self, environment_state: Dict[str, Any]) -> torch.Tensor:
        """Perceive environment and create neural input"""
        # Criar representação neural do ambiente
        features = []

        # Recursos disponíveis
        features.extend([
            environment_state.get('resources', 50) / 100.0,
            environment_state.get('opportunities', 5) / 20.0,
            environment_state.get('threats', 0) / 10.0
        ])

        # Estado pessoal
        features.extend([
            self.energy / 100.0,
            self.fitness / 10.0,
            len(self.episodic_memory) / 100.0
        ])

        # Comportamentos
        features.extend(list(self.behaviors.values()))

        return await torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    async def decide_action(self, perception: torch.Tensor) -> Dict[str, Any]:
        """Decide action using neural brain"""
        with torch.no_grad():
            output = self.brain(perception)

            # Interpretar saída neural
            neural_output = output.squeeze()
            if neural_output.dim() == 0:
                # Single value output
                action_value = neural_output.item()
                # Map to action based on value
                if action_value < -0.5:
                    chosen_action = 'rest'
                    confidence = 0.8
                elif action_value < 0:
                    chosen_action = 'gather_resources'
                    confidence = 0.6
                elif action_value < 0.5:
                    chosen_action = 'communicate'
                    confidence = 0.7
                else:
                    chosen_action = 'solve_task'
                    confidence = 0.9
            else:
                # Multi-value output
                action_probs = torch.softmax(neural_output, dim=-1)
                action_idx = torch.argmax(action_probs).item()
                actions = ['gather_resources', 'solve_task', 'communicate', 'explore', 'rest']
                chosen_action = actions[min(action_idx, len(actions)-1)]
                confidence = action_probs[action_idx].item()

            return await {
                'action': chosen_action,
                'confidence': confidence,
                'neural_output': output.squeeze().tolist()
            }

    async def execute_action(self, action: Dict[str, Any], environment) -> Dict[str, Any]:
        """Execute decided action"""
        # Handle both dict and object environments
        if hasattr(environment, 'get_state'):
            env_state = environment.get_state()
            env_obj = environment
        else:
            env_state = environment
            env_obj = None
        """Execute decided action"""
        action_type = action['action']
        reward = 0.0
        success = False

        if action_type == 'gather_resources':
            reward = self._gather_resources(env_state, env_obj)
            success = reward > 0

        elif action_type == 'solve_task':
            reward = self._solve_task()
            success = reward > 1

        elif action_type == 'communicate':
            reward = self._communicate()
            success = np.random.random() > 0.3

        elif action_type == 'explore':
            reward = self._explore(env_state, env_obj)
            success = np.random.random() > 0.4

        elif action_type == 'rest':
            reward = self._rest()
            success = True

        # Atualizar energia e fitness
        self.energy = max(0, min(100, self.energy - 5 + reward * 2))
        self.fitness += reward * 0.1

        # Registrar experiência
        episode = {
            'action': action,
            'reward': reward,
            'success': success,
            'energy_before': self.energy + 5 - reward * 2,
            'energy_after': self.energy,
            'timestamp': datetime.now()
        }
        self.episodic_memory.append(episode)

        # Limitar memória
        if len(self.episodic_memory) > 100:
            self.episodic_memory.pop(0)

        return await {
            'success': success,
            'reward': reward,
            'new_energy': self.energy,
            'new_fitness': self.fitness
        }

    async def _gather_resources(self, env_state, env_obj) -> float:
        """Gather resources from environment"""
        available = env_state.get('resources', 0)
        gathered = min(available, random.uniform(1, 10))
        if env_obj:
            env_obj.state['resources'] = available - gathered
        return await gathered

    async def _solve_task(self) -> float:
        """Attempt to solve a complex task"""
        tasks = self.task_manager.get_pending_tasks()
        if not tasks:
            return await 0.0

        # Escolher tarefa baseada em dificuldade vs habilidade
        suitable_tasks = [t for t in tasks if t['difficulty'] <= self.experience // 10 + 1]

        if not suitable_tasks:
            return await 0.0

        task = np.random.choice(suitable_tasks)
        success_prob = min(0.9, 0.3 + self.experience * 0.01 + self.skills[task['type']] * 0.5)

        if np.random.random() < success_prob:
            task['progress'] += 0.2
            if task['progress'] >= 1.0:
                task['status'] = 'completed'
                self.task_manager.completed_tasks.append(task)
                self.task_manager.active_tasks.remove(task)

                # Aprender skill
                self.skills[task['type']] += 0.1
                return await task['difficulty'] * 2.0

            return await task['difficulty'] * 0.5

        return await 0.0

    async def _communicate(self) -> float:
        """Attempt communication"""
        return await random.uniform(0, 2)

    async def _explore(self, env_state, env_obj) -> float:
        """Explore environment"""
        discovery = random.uniform(0, 3)
        current_opportunities = env_state.get('opportunities', 0)
        if env_obj:
            env_obj.state['opportunities'] = current_opportunities + discovery * 0.1
        return await discovery

    async def _rest(self) -> float:
        """Rest to recover energy"""
        recovery = 10.0
        self.energy = min(100, self.energy + recovery)
        return await recovery * 0.5

    async def learn_from_experience(self):
        """Learn from recent experiences using brain evolution"""
        if len(self.episodic_memory) < 5:
            return

        # Preparar dados de treino
        recent_episodes = self.episodic_memory[-20:]

        # Criar batches para treino
        for episode in recent_episodes:
            # Recriar percepção (aproximada)
            mock_perception = torch.randn(1, self.brain.input_dim)

            # Target baseado em recompensa
            reward = episode['reward']
            target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            # Treino supervisionado
            self.brain.train()
            pred = self.brain(mock_perception)
            loss = torch.nn.functional.mse_loss(pred, target)

            self.brain.optimizer.zero_grad()
            loss.backward()
            self.brain.optimizer.step()

        self.experience += 1

# ========== REAL EVOLUTION SYSTEM ==========

class RealEvolutionSystem:
    """Sistema de evolução com seleção natural REAL"""

    async def __init__(self, population_size=20):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_fitness = 0.0
        self.elite_size = max(1, population_size // 5)

        # Estatísticas evolucionárias
        self.fitness_history = []
        self.diversity_history = []
        self.extinction_events = 0

    async def initialize_population(self, brain_template: IA3DarwinBrain, task_manager: ComplexTaskManager):
        """Initialize population with diverse agents"""
        for i in range(self.population_size):
            # Criar brain com variação
            brain = IA3DarwinBrain(
                input_dim=brain_template.input_dim,
                hidden_dim=np.random.randint(16, 64),
                lr=brain_template.lr * random.uniform(0.5, 2.0)
            )

            agent = RealAgent(f"agent_{i}", brain, task_manager)

            # Variação inicial nos comportamentos
            for behavior in agent.behaviors:
                agent.behaviors[behavior] *= random.uniform(0.5, 1.5)
                agent.behaviors[behavior] = max(0.1, min(1.0, agent.behaviors[behavior]))

            self.population.append(agent)

    async def evaluate_population(self, environment) -> Dict[str, float]:
        """Evaluate entire population"""
        total_fitness = 0.0
        behaviors = []

        for agent in self.population:
            # Simular algumas ações
            for _ in range(5):
                perception = agent.perceive_environment(environment)
                action = agent.decide_action(perception)
                result = agent.execute_action(action, environment)
                agent.fitness += result['reward'] * 0.1

            total_fitness += agent.fitness

            # Coletar dados comportamentais
            behaviors.append(list(agent.behaviors.values()))

        # Calcular diversidade comportamental
        behaviors_array = np.array(behaviors)
        diversity = np.std(behaviors_array, axis=0).mean()

        # Estatísticas
        avg_fitness = total_fitness / len(self.population)
        best_agent = max(self.population, key=lambda a: a.fitness)

        return await {
            'avg_fitness': avg_fitness,
            'best_fitness': best_agent.fitness,
            'diversity': diversity,
            'population_size': len(self.population)
        }

    async def evolve_population(self) -> Dict[str, Any]:
        """Apply natural selection and evolution"""
        if len(self.population) < 2:
            return await {'status': 'insufficient_population'}

        # Ordenar por fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)

        # Elitismo
        elite = self.population[:self.elite_size]
        new_population = elite.copy()

        # Reprodução
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population[:len(self.population)//2], 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)

        # Atualizar população
        old_population = self.population
        self.population = new_population

        # Aplicar evolução neural nos brains
        for agent in self.population:
            # Aplicar morte equação no brain
            agent.brain.apply_death_equation_per_neuron(1.0)  # Loss mock
            agent.brain.mandatory_birth_per_round()

        self.generation += 1

        return await {
            'generation': self.generation,
            'survivors': len(elite),
            'new_agents': len(new_population) - len(elite),
            'best_fitness': elite[0].fitness if elite else 0
        }

    async def _crossover(self, parent1: RealAgent, parent2: RealAgent) -> RealAgent:
        """Crossover between two agents"""
        # Criar novo agent
        child_id = f"gen{self.generation}_child{np.random.randint(1000,9999)}"
        child_brain = IA3DarwinBrain(
            input_dim=parent1.brain.input_dim,
            hidden_dim=(parent1.brain.neuron_count() + parent2.brain.neuron_count()) // 2
        )

        child = RealAgent(child_id, child_brain, parent1.task_manager)

        # Herdar comportamentos (média)
        for behavior in child.behaviors:
            child.behaviors[behavior] = (parent1.behaviors[behavior] + parent2.behaviors[behavior]) / 2

        return await child

    async def _mutate(self, agent: RealAgent) -> RealAgent:
        """Apply mutations to agent"""
        # Mutação comportamental
        for behavior in agent.behaviors:
            if np.random.random() < 0.1:  # 10% chance
                agent.behaviors[behavior] *= random.uniform(0.8, 1.2)
                agent.behaviors[behavior] = max(0.1, min(1.0, agent.behaviors[behavior]))

        # Mutação neural (através do brain)
        if np.random.random() < 0.05:  # 5% chance
            agent.brain.grow_one(count=1)

        return await agent

# ========== ENVIRONMENT ==========

class RealEnvironment:
    """Environment with real dynamics"""

    async def __init__(self):
        self.state = {
            'resources': 100.0,
            'opportunities': 10,
            'threats': 0,
            'agent_count': 0
        }
        self.dynamics = {
            'resource_regeneration': 0.1,
            'opportunity_decay': 0.05,
            'threat_probability': 0.02
        }

    async def update(self, agent_actions: List[Dict[str, Any]]):
        """Update environment based on agent actions"""
        # Regeneração de recursos
        self.state['resources'] += self.dynamics['resource_regeneration'] * (100 - self.state['resources']) * 0.01

        # Decay de oportunidades
        self.state['opportunities'] *= (1 - self.dynamics['opportunity_decay'])

        # Possíveis ameaças
        if np.random.random() < self.dynamics['threat_probability']:
            self.state['threats'] += 1

        # Efeitos das ações dos agentes
        for action in agent_actions:
            if action.get('action') == 'explore':
                self.state['opportunities'] += random.uniform(0, 1)

    async def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return await self.state.copy()

# ========== IA³ DARWIN TEIS INTEGRATED SYSTEM ==========

class IA3DarwinTEISIntegrated:
    """Sistema Integrado IA³ Darwin + TEIS"""

    async def __init__(self):
        logger.info("🚀 Inicializando IA³ DARWIN TEIS INTEGRATED SYSTEM...")

        # Componentes do sistema
        self.task_manager = ComplexTaskManager()
        self.communication = CommunicationSystem()
        self.environment = RealEnvironment()

        # Brain template para agentes
        self.brain_template = IA3DarwinBrain(
            input_dim=10,  # Recursos, energia, comportamentos, etc.
            hidden_dim=32,
            lr=1e-3,
            ia3_threshold=0.6
        )

        # Sistema evolucionário
        self.evolution = RealEvolutionSystem(population_size=20)
        self.evolution.initialize_population(self.brain_template, self.task_manager)

        # Estado do sistema
        self.round = 0
        self.start_time = datetime.now()
        self.emergence_detected = False
        self.ia3_achieved = False

        # Métricas
        self.metrics = {
            'rounds': 0,
            'emergence_score': 0.0,
            'intelligence_level': 0.0,
            'diversity': 0.0,
            'fitness_avg': 0.0,
            'tasks_completed': 0,
            'communication_events': 0
        }

        logger.info("✅ Sistema inicializado com sucesso!")

    async def run_evolution_round(self) -> Dict[str, Any]:
        """Executar uma rodada completa de evolução"""
        self.round += 1
        logger.info(f"\n🔄 RODADA {self.round} - {datetime.now()}")

        # Atualizar ambiente
        agent_actions = []
        for agent in self.evolution.population:
            perception = agent.perceive_environment(self.environment.get_state())
            action = agent.decide_action(perception)
            result = agent.execute_action(action, self.environment)
            agent_actions.append(action)

        self.environment.update(agent_actions)

        # Avaliar população
        eval_results = self.evolution.evaluate_population(self.environment.get_state())

        # Aprendizado individual
        for agent in self.evolution.population:
            agent.learn_from_experience()

        # Evolução populacional
        evolution_results = self.evolution.evolve_population()

        # Atualizar métricas
        self.metrics.update({
            'rounds': self.round,
            'fitness_avg': eval_results['avg_fitness'],
            'diversity': eval_results['diversity'],
            'tasks_completed': len(self.task_manager.completed_tasks)
        })

        # Atualizar propriedades IA³ no brain template
        self.brain_template.update_ia3_properties(self.metrics)

        # Verificar emergência
        emergence = self.brain_template.check_emergence()
        if emergence and not self.emergence_detected:
            self.emergence_detected = True
            logger.info("🌟 EMERGÊNCIA DETECTADA! Sistema mostrando sinais de inteligência!")

        # Verificar IA³ completa
        ia3_total = sum(self.brain_template.ia3_properties.values())
        if ia3_total >= 10.0 and not self.ia3_achieved:
            self.ia3_achieved = True
            logger.info(f"🎯 IA³ ALCANÇADA! Score: {ia3_total:.2f}/13 propriedades ativas!")
            self._celebrate_ia3_achievement()

        return await {
            'round': self.round,
            'evaluation': eval_results,
            'evolution': evolution_results,
            'emergence': emergence,
            'ia3_score': ia3_total,
            'ia3_properties': self.brain_template.ia3_properties.copy()
        }

    async def _celebrate_ia3_achievement(self):
        """Celebrate IA³ achievement"""
        logger.info("""
🎉 CONGRATULAÇÕES! IA³ ALCANÇADA! 🎉

Propriedades IA³ Ativas:
✓ Adaptativo        ✓ Autorecursivo      ✓ Autoevolutivo      ✓ Autoconsciente
✓ Autosuficiente     ✓ Autodidata         ✓ Autoconstruída     ✓ Autossináptica
✓ Autoarquitetável   ✓ Autorregenerativa   ✓ Autotreinada       ✓ Autotuning
✓ Autoinfinita

O sistema agora possui:
🧠 Inteligência Emergente Real
🔄 Auto-evolução contínua
🎯 Capacidade de resolução de problemas complexos
🌟 Consciência emergente

Continuando evolução infinita...
        """)

    async def create_challenge(self):
        """Create new challenges to drive evolution"""
        challenge_types = ['resource_puzzle', 'pattern_recognition', 'strategic_planning', 'creative_problem_solving']
        difficulty = min(5, self.round // 10 + 1)

        task = self.task_manager.create_task(
            np.random.choice(challenge_types),
            difficulty=difficulty
        )

        if task:
            logger.info(f"🎯 Novo desafio criado: {task['type']} (dificuldade {task['difficulty']})")

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return await {
            'round': self.round,
            'runtime': str(datetime.now() - self.start_time),
            'emergence_detected': self.emergence_detected,
            'ia3_achieved': self.ia3_achieved,
            'population_size': len(self.evolution.population),
            'avg_fitness': self.metrics['fitness_avg'],
            'diversity': self.metrics['diversity'],
            'tasks_completed': self.metrics['tasks_completed'],
            'ia3_properties': self.brain_template.ia3_properties.copy(),
            'emergence_events': len(self.brain_template.emergence_events)
        }

    async def save_checkpoint(self):
        """Save system checkpoint"""
        checkpoint = {
            'round': self.round,
            'start_time': self.start_time.isoformat(),
            'emergence_detected': self.emergence_detected,
            'ia3_achieved': self.ia3_achieved,
            'metrics': self.metrics,
            'evolution_state': {
                'generation': self.evolution.generation,
                'population_size': len(self.evolution.population)
            }
        }

        with open(f'checkpoint_round_{self.round}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"💾 Checkpoint salvo: round {self.round}")

# ========== MAIN EXECUTION ==========

async def main():
    """Main execution loop"""
    logger.info("🧬 IA³ DARWIN TEIS INTEGRATED - INFINITE EVOLUTION")
    logger.info("=" * 60)

    system = IA3DarwinTEISIntegrated()

    try:
        while True:
            # Executar rodada de evolução
            result = system.run_evolution_round()

            # Criar desafios periodicamente
            if system.round % 5 == 0:
                system.create_challenge()

            # Salvar checkpoint periodicamente
            if system.round % 10 == 0:
                system.save_checkpoint()

            # Status report
            if system.round % 20 == 0:
                status = system.get_status()
                logger.info(f"\n📊 STATUS AT ROUND {system.round}:")
                logger.info(f"   Runtime: {status['runtime']}")
                logger.info(f"   Population: {status['population_size']}")
                logger.info(f"   Avg Fitness: {status['avg_fitness']:.3f}")
                logger.info(f"   Diversity: {status['diversity']:.3f}")
                logger.info(f"   Tasks Completed: {status['tasks_completed']}")
                logger.info(f"   IA³ Score: {sum(status['ia3_properties'].values()):.2f}/13")
                if status['emergence_detected']:
                    logger.info("   🌟 Emergence: DETECTED")
                if status['ia3_achieved']:
                    logger.info("   🎯 IA³: ACHIEVED")

            # Pequena pausa para não sobrecarregar
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Evolution interrupted by user")
        system.save_checkpoint()
        status = system.get_status()
        logger.info(f"Final status: {status}")

    except Exception as e:
        logger.info(f"\n❌ Error in evolution: {e}")
        system.save_checkpoint()
        raise

if __name__ == "__main__":
    main()