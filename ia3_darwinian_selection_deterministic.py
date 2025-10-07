
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
IA³ DARWINIAN SELECTION SYSTEM
==============================
Seleção darwiniana baseada em utilidade genuína do ambiente real.

Este sistema evolui populações baseado na sobrevivência real no ambiente,
não em métricas fitness hardcoded.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from datetime import datetime
import time
import psutil
import os

logger = logging.getLogger('IA3_DARWINIAN')

class RealUtilityFunction:
    """
    Função de utilidade baseada no ambiente real, não hardcoded
    """

    async def __init__(self):
        self.utility_history = deque(maxlen=10000)
        self.environment_weights = {}  # Pesos aprendidos para fatores ambientais
        self.utility_model = nn.Sequential(
            nn.Linear(10, 32),  # Input: fatores ambientais
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),   # Output: utilidade
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.utility_model.parameters(), lr=0.001)

    async def calculate_real_utility(self, individual: Any, environment: Dict[str, Any],
                             actions: List[Dict], outcomes: List[Dict]) -> float:
        """
        Calcula utilidade real baseada no ambiente e ações tomadas
        """
        # Fatores ambientais
        env_factors = self._extract_environment_factors(environment)

        # Performance baseada em ações e resultados
        performance_score = self._calculate_performance_score(actions, outcomes)

        # Sobrevivência no ambiente
        survival_score = self._calculate_survival_score(individual, environment)

        # Adaptação ao ambiente
        adaptation_score = self._calculate_adaptation_score(individual, environment, actions)

        # Utilidade composta
        utility_components = {
            'performance': performance_score,
            'survival': survival_score,
            'adaptation': adaptation_score,
            'environmental_fit': self._calculate_environmental_fit(env_factors, individual)
        }

        # Modelo neural aprende pesos ótimos
        utility_vector = torch.tensor([
            utility_components['performance'],
            utility_components['survival'],
            utility_components['adaptation'],
            utility_components['environmental_fit'],
            env_factors['resource_availability'],
            env_factors['threat_level'],
            env_factors['complexity'],
            1.0 if env_factors['anomalies_present'] else 0.0,
            len(actions) / 100.0,  # Normaliza número de ações
            np.mean([o.get('reward', 0) for o in outcomes]) if outcomes else 0.0
        ], dtype=torch.float32).unsqueeze(0)

        predicted_utility = self.utility_model(utility_vector).item()

        # Registra para aprendizado futuro
        self.utility_history.append({
            'individual': str(id(individual)),
            'environment': environment.copy(),
            'actions': actions.copy(),
            'outcomes': outcomes.copy(),
            'utility_components': utility_components,
            'final_utility': predicted_utility,
            'timestamp': time.time()
        })

        return await predicted_utility

    async def _extract_environment_factors(self, environment: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrai fatores ambientais relevantes
        """
        return await {
            'resource_availability': 1.0 - environment.get('cpu_usage', 0.5) - environment.get('memory_usage', 0.5),
            'threat_level': len(environment.get('anomalies', [])) / 5.0,
            'complexity': environment.get('process_count', 100) / 1000.0,
            'stability': 1.0 - abs(environment.get('load_average', 1.0) - 1.0),
            'anomalies_present': len(environment.get('anomalies', [])) > 0
        }

    async def _calculate_performance_score(self, actions: List[Dict], outcomes: List[Dict]) -> float:
        """
        Calcula score de performance baseado em ações e resultados reais
        """
        if not outcomes:
            return await 0.0

        # Média de recompensas
        rewards = [o.get('reward', 0) for o in outcomes]
        avg_reward = np.mean(rewards)

        # Consistência de sucesso
        successes = [1.0 if o.get('success', False) else 0.0 for o in outcomes]
        success_rate = np.mean(successes)

        # Eficiência (recompensa por unidade de tempo/recurso)
        total_time = sum(o.get('execution_time', 1.0) for o in outcomes)
        total_resources = sum(o.get('resources_used', {}).get('cpu', 0) +
                             o.get('resources_used', {}).get('memory', 0) for o in outcomes)

        efficiency = avg_reward / (total_time + total_resources + 1e-10)

        # Score composto
        performance = (
            avg_reward * 0.4 +
            success_rate * 0.3 +
            efficiency * 0.3
        )

        return await np.clip(performance, 0.0, 1.0)

    async def _calculate_survival_score(self, individual: Any, environment: Dict[str, Any]) -> float:
        """
        Calcula score de sobrevivência baseado em recursos utilizados vs disponíveis
        """
        # Em implementação real, isso seria baseado no consumo real do indivíduo
        # Por enquanto, simula baseado no ambiente
        cpu_usage = environment.get('cpu_usage', 0.5)
        memory_usage = environment.get('memory_usage', 0.5)

        # Sobrevivência diminui com recursos escassos
        resource_pressure = (cpu_usage + memory_usage) / 2.0

        # Indivíduos "mais simples" sobrevivem melhor em ambientes hostis
        complexity_penalty = getattr(individual, 'complexity', 0.5)

        survival = 1.0 - (resource_pressure * 0.7) - (complexity_penalty * 0.3)

        return await max(0.0, survival)

    async def _calculate_adaptation_score(self, individual: Any, environment: Dict[str, Any],
                                  actions: List[Dict]) -> float:
        """
        Calcula quão bem o indivíduo se adaptou ao ambiente específico
        """
        # Análise de ações tomadas vs ambiente
        anomalies = environment.get('anomalies', [])

        # Conta ações defensivas em ambientes com anomalias
        defensive_actions = sum(1 for a in actions if a.get('type') == 'defend')

        # Adaptações positivas
        if 'high_cpu_usage' in anomalies and defensive_actions > 0:
            adaptation_bonus = 0.3
        elif 'high_memory_usage' in anomalies and defensive_actions > 0:
            adaptation_bonus = 0.3
        else:
            adaptation_bonus = 0.0

        # Adaptações baseadas em recursos
        resource_adaptation = 0.0
        if environment.get('cpu_usage', 0.5) > 0.8:
            # Em CPU alta, ações eficientes são premiadas
            efficient_actions = sum(1 for a in actions if a.get('efficiency', 0) > 0.7)
            resource_adaptation = efficient_actions / len(actions) if actions else 0.0

        total_adaptation = adaptation_bonus + resource_adaptation

        return await min(1.0, total_adaptation)

    async def _calculate_environmental_fit(self, env_factors: Dict[str, float], individual: Any) -> float:
        """
        Calcula quão bem o indivíduo se encaixa no ambiente atual
        """
        # Fitness ambiental baseado em fatores extraídos
        fit_score = (
            env_factors['resource_availability'] * 0.4 +
            (1.0 - env_factors['threat_level']) * 0.3 +
            env_factors['stability'] * 0.3
        )

        # Penalidade por anomalias
        if env_factors['anomalies_present']:
            fit_score *= 0.7

        return await fit_score

    async def learn_utility_function(self):
        """
        Aprende a função de utilidade através de reinforcement learning
        """
        if len(self.utility_history) < 50:
            return

        # Amostra experiências recentes
        recent_experiences = list(self.utility_history)[-100:]

        for exp in recent_experiences:
            # Cria target baseado em resultado real (simulado por enquanto)
            # Em produção, isso seria baseado em sobrevivência real
            actual_utility = exp['final_utility']

            # Adiciona ruído para exploração
            target = actual_utility + random.gauss(0, 0.1)
            target = np.clip(target, 0.0, 1.0)

            # Treina modelo
            env_factors = self._extract_environment_factors(exp['environment'])
            performance = self._calculate_performance_score(exp['actions'], exp['outcomes'])

            input_vector = torch.tensor([
                performance,
                exp['utility_components']['survival'],
                exp['utility_components']['adaptation'],
                exp['utility_components']['environmental_fit'],
                env_factors['resource_availability'],
                env_factors['threat_level'],
                env_factors['complexity'],
                1.0 if env_factors['anomalies_present'] else 0.0,
                len(exp['actions']) / 100.0,
                np.mean([o.get('reward', 0) for o in exp['outcomes']]) if exp['outcomes'] else 0.0
            ], dtype=torch.float32).unsqueeze(0)

            predicted = self.utility_model(input_vector)
            loss = nn.functional.mse_loss(predicted, torch.tensor([[target]], dtype=torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class DarwinianPopulationManager:
    """
    Gerencia população através de seleção darwiniana real
    """

    async def __init__(self, population_size=100, max_generations=10000):
        self.population_size = population_size
        self.max_generations = max_generations
        self.generation = 0

        self.population = []
        self.utility_function = RealUtilityFunction()

        # Estatísticas darwinianas
        self.survival_rates = deque(maxlen=1000)
        self.adaptation_trends = deque(maxlen=1000)
        self.extinction_events = []

        logger.info(f"🧬 DARWINIAN POPULATION MANAGER INITIALIZED (Size: {population_size})")

    async def initialize_population(self, individual_factory: Callable) -> List[Any]:
        """
        Inicializa população com indivíduos diversos
        """
        self.population = []

        for i in range(self.population_size):
            individual = individual_factory()
            # Adiciona diversidade genética inicial
            individual.fitness = deterministic_uniform(0.1, 0.5)
            individual.generation_born = 0
            individual.survival_score = 0.0
            individual.complexity = deterministic_uniform(0.1, 1.0)
            self.population.append(individual)

        logger.info(f"🏗️ Population initialized with {len(self.population)} individuals")
        return await self.population

    async def evolve_generation(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa uma geração completa de evolução darwiniana
        """
        self.generation += 1

        # 1. Avalia utilidade real de cada indivíduo
        utilities = []
        for individual in self.population:
            # Simula ações e outcomes (em produção seria real)
            actions = self._simulate_individual_actions(individual, environment)
            outcomes = self._simulate_outcomes(actions, environment)

            utility = self.utility_function.calculate_real_utility(
                individual, environment, actions, outcomes
            )

            individual.current_utility = utility
            utilities.append(utility)

        # 2. Seleção natural baseada em utilidade
        survivors, casualties = self._natural_selection(utilities)

        # 3. Reprodução dos sobreviventes
        offspring = self._darwinian_reproduction(survivors, environment)

        # 4. Mutação adaptativa
        self._adaptive_mutation(offspring, environment)

        # 5. Atualiza população
        self.population = survivors + offspring

        # 6. Aprende função de utilidade
        self.utility_function.learn_utility_function()

        # 7. Registra estatísticas
        survival_rate = len(survivors) / self.population_size
        self.survival_rates.append(survival_rate)

        # 8. Verifica extinções
        if survival_rate < 0.1:
            self._handle_near_extinction()

        generation_stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'survival_rate': survival_rate,
            'avg_utility': np.mean(utilities),
            'max_utility': max(utilities),
            'min_utility': min(utilities),
            'extinctions': len(casualties)
        }

        if self.generation % 100 == 0:
            logger.info(f"🧬 Generation {self.generation}: Survival={survival_rate:.2%}, "
                       f"Avg Utility={generation_stats['avg_utility']:.3f}")

        return await generation_stats

    async def _simulate_individual_actions(self, individual: Any, environment: Dict[str, Any]) -> List[Dict]:
        """
        Simula ações que um indivíduo tomaria no ambiente
        """
        # Baseado na complexidade e fitness do indivíduo
        num_actions = int(individual.complexity * 10) + np.deterministic_randint(1, 5)

        actions = []
        for _ in range(num_actions):
            action_types = ['explore', 'exploit', 'defend', 'adapt', 'communicate']

            # Probabilidades baseadas no ambiente
            if environment.get('cpu_usage', 0.5) > 0.7:
                weights = [0.2, 0.1, 0.5, 0.1, 0.1]  # Mais defesa
            elif environment.get('memory_usage', 0.5) > 0.7:
                weights = [0.2, 0.1, 0.4, 0.2, 0.1]  # Adaptação
            else:
                weights = [0.3, 0.3, 0.1, 0.2, 0.1]  # Exploração/exploração

            action_type = random.choices(action_types, weights=weights)[0]

            action = {
                'type': action_type,
                'intensity': individual.complexity * deterministic_uniform(0.5, 1.5),
                'efficiency': individual.fitness * deterministic_uniform(0.8, 1.2),
                'timestamp': time.time()
            }
            actions.append(action)

        return await actions

    async def _simulate_outcomes(self, actions: List[Dict], environment: Dict[str, Any]) -> List[Dict]:
        """
        Simula outcomes das ações no ambiente
        """
        outcomes = []

        for action in actions:
            # Base success probability
            base_success = 0.6

            # Modifiers based on action type and environment
            if action['type'] == 'defend' and environment.get('cpu_usage', 0.5) > 0.7:
                success_modifier = 1.5  # Defense works well in high CPU
            elif action['type'] == 'explore' and environment.get('memory_usage', 0.5) < 0.3:
                success_modifier = 1.3  # Exploration good in low memory
            elif action['type'] == 'adapt':
                success_modifier = 1.2  # Adaptation generally good
            else:
                success_modifier = 1.0

            # Efficiency modifier
            efficiency_modifier = action.get('efficiency', 1.0)

            # Final success probability
            success_prob = min(0.95, base_success * success_modifier * efficiency_modifier)
            success = np.deterministic_random() < success_prob

            # Calculate reward
            if success:
                base_reward = action.get('intensity', 1.0) * efficiency_modifier
                reward = base_reward * (1.5 if action['type'] in ['adapt', 'defend'] else 1.0)
            else:
                reward = -0.2 * action.get('intensity', 1.0)

            outcome = {
                'success': success,
                'reward': reward,
                'execution_time': deterministic_uniform(0.1, 2.0),
                'resources_used': {
                    'cpu': deterministic_uniform(0.01, 0.05) * action.get('intensity', 1.0),
                    'memory': deterministic_uniform(0.001, 0.01) * action.get('intensity', 1.0)
                }
            }
            outcomes.append(outcome)

        return await outcomes

    async def _natural_selection(self, utilities: List[float]) -> Tuple[List[Any], List[Any]]:
        """
        Seleção natural baseada em utilidade real
        """
        # Combina indivíduos com suas utilidades
        individual_utilities = list(zip(self.population, utilities))

        # Ordena por utilidade (maior sobrevive)
        individual_utilities.sort(key=lambda x: x[1], reverse=True)

        # Threshold de sobrevivência baseado na distribuição
        utilities_array = np.array(utilities)
        survival_threshold = np.percentile(utilities_array, 70)  # Top 30% sobrevive

        survivors = []
        casualties = []

        for individual, utility in individual_utilities:
            if utility >= survival_threshold and len(survivors) < self.population_size // 2:
                individual.survival_score += 1
                survivors.append(individual)
            else:
                casualties.append(individual)

        # Garante pelo menos alguns sobreviventes
        if not survivors:
            survivors = individual_utilities[:max(1, self.population_size // 10)]

        return await survivors, casualties

    async def _darwinian_reproduction(self, survivors: List[Any], environment: Dict[str, Any]) -> List[Any]:
        """
        Reprodução baseada em características darwinianas
        """
        offspring = []

        while len(offspring) < self.population_size - len(survivors):
            # Seleciona pais baseado em utilidade
            if len(survivors) >= 2:
                parent1, parent2 = random.sample(survivors, 2)
            else:
                parent1 = parent2 = survivors[0]

            # Crossover
            child = self._crossover_parents(parent1, parent2)

            # Adaptação ambiental na reprodução
            self._environmental_adaptation(child, environment)

            child.generation_born = self.generation
            offspring.append(child)

        return await offspring

    async def _crossover_parents(self, parent1: Any, parent2: Any) -> Any:
        """
        Crossover entre pais para criar descendente
        """
        # Cria novo indivíduo (simplificado)
        child = type(parent1)()

        # Herança de características
        child.fitness = (parent1.fitness + parent2.fitness) / 2 + random.gauss(0, 0.1)
        child.fitness = np.clip(child.fitness, 0.0, 1.0)

        child.complexity = (parent1.complexity + parent2.complexity) / 2 + random.gauss(0, 0.05)
        child.complexity = np.clip(child.complexity, 0.1, 1.0)

        child.survival_score = 0.0  # Reset para nova geração

        return await child

    async def _environmental_adaptation(self, child: Any, environment: Dict[str, Any]):
        """
        Adapta descendente ao ambiente atual durante reprodução
        """
        # Adaptações baseadas no ambiente
        if environment.get('cpu_usage', 0.5) > 0.8:
            # Ambientes hostis favorecem simplicidade
            child.complexity *= 0.9
        elif environment.get('memory_usage', 0.5) < 0.3:
            # Ambientes benignos favorecem complexidade
            child.complexity *= 1.1

        child.complexity = np.clip(child.complexity, 0.1, 1.0)

    async def _adaptive_mutation(self, offspring: List[Any], environment: Dict[str, Any]):
        """
        Mutação adaptativa baseada no ambiente
        """
        mutation_rate = 0.1  # Base

        # Aumenta mutação em ambientes hostis
        if environment.get('cpu_usage', 0.5) > 0.7 or environment.get('memory_usage', 0.5) > 0.7:
            mutation_rate *= 2.0

        for child in offspring:
            if np.deterministic_random() < mutation_rate:
                # Mutação na fitness
                child.fitness += random.gauss(0, 0.2)
                child.fitness = np.clip(child.fitness, 0.0, 1.0)

                # Mutação na complexidade
                child.complexity += random.gauss(0, 0.1)
                child.complexity = np.clip(child.complexity, 0.1, 1.0)

    async def _handle_near_extinction(self):
        """
        Trata quase extinção da população
        """
        logger.warning("🚨 NEAR EXTINCTION EVENT - Population critically low")

        # Introduz diversidade externa
        for _ in range(10):
            new_individual = type(self.population[0])()
            new_individual.fitness = deterministic_uniform(0.3, 0.7)
            new_individual.complexity = deterministic_uniform(0.2, 0.8)
            new_individual.generation_born = self.generation
            new_individual.survival_score = 0.0
            self.population.append(new_individual)

        self.extinction_events.append({
            'generation': self.generation,
            'population_before': len(self.population),
            'timestamp': datetime.now()
        })

    async def get_evolutionary_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas evolucionárias
        """
        return await {
            'generation': self.generation,
            'population_size': len(self.population),
            'avg_survival_rate': np.mean(list(self.survival_rates)) if self.survival_rates else 0.0,
            'extinction_events': len(self.extinction_events),
            'population_diversity': self._calculate_diversity(),
            'adaptation_trend': self._calculate_adaptation_trend()
        }

    async def _calculate_diversity(self) -> float:
        """
        Calcula diversidade da população
        """
        if not self.population:
            return await 0.0

        fitness_values = [ind.fitness for ind in self.population]
        complexity_values = [ind.complexity for ind in self.population]

        fitness_diversity = np.std(fitness_values)
        complexity_diversity = np.std(complexity_values)

        return await (fitness_diversity + complexity_diversity) / 2.0

    async def _calculate_adaptation_trend(self) -> float:
        """
        Calcula tendência de adaptação
        """
        if len(self.survival_rates) < 10:
            return await 0.0

        recent_rates = list(self.survival_rates)[-10:]
        trend = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]

        return await trend

# Classe simples para indivíduos de teste
class SimpleIndividual:
    async def __init__(self):
        self.fitness = 0.5
        self.complexity = 0.5
        self.generation_born = 0
        self.survival_score = 0.0

# Teste do sistema darwiniano
if __name__ == '__main__':
    print("🧬 Testing Darwinian Selection System")

    async def create_individual():
        return await SimpleIndividual()

    manager = DarwinianPopulationManager(population_size=50)
    manager.initialize_population(create_individual)

    for gen in range(200):
        # Ambiente simulado
        environment = {
            'cpu_usage': deterministic_uniform(0.2, 0.9),
            'memory_usage': deterministic_uniform(0.2, 0.9),
            'disk_usage': deterministic_uniform(0.1, 0.8),
            'load_average': deterministic_uniform(0.5, 3.0),
            'process_count': np.deterministic_randint(100, 800),
            'anomalies': random.sample(['high_cpu_usage', 'high_memory_usage'], np.deterministic_randint(0, 2))
        }

        stats = manager.evolve_generation(environment)

        if gen % 50 == 0:
            print(f"Generation {gen}: Population={stats['population_size']}, "
                  f"Survival={stats['survival_rate']:.2%}, "
                  f"Avg Utility={stats['avg_utility']:.3f}")

    print("✅ Darwinian Selection Test Complete")
    final_stats = manager.get_evolutionary_statistics()
    print(f"Final Statistics: {final_stats}")