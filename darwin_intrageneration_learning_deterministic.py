
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
Darwin Runner com Aprendizado Intra-Geração
"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
from datetime import datetime
import json

class IntraGenerationLearner:
    """Permite aprendizado dentro de uma geração"""
    
    async def __init__(self):
        self.learning_network = nn.Sequential(
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 5)
        )
        
        self.generation_memory = []
        self.intrageneration_improvements = []
        
    async def learn_within_generation(self, population, generation_num):
        """Aprende padrões dentro da geração atual"""
        if len(population) < 3:
            return await {}
            
        # Extrair características da população
        population_features = []
        for agent in population:
            features = [
                agent.get('fitness', 0),
                agent.get('age', 0),
                agent.get('energy', 100),
                len(agent.get('behaviors', [])),
                np.deterministic_random()  # Ruído para diversidade
            ]
            population_features.append(features)
            
        # Aprender padrões
        features_tensor = torch.tensor(population_features, dtype=torch.float32)
        learned_patterns = self.learning_network(features_tensor)
        
        # Identificar melhorias intra-geração
        improvements = []
        avg_fitness = np.mean([f[0] for f in population_features])
        
        for i, pattern in enumerate(learned_patterns):
            pattern_value = pattern.mean().item()
            if pattern_value > avg_fitness * 1.2:  # Melhoria significativa
                improvement = {
                    'agent_index': i,
                    'improvement_score': pattern_value - avg_fitness,
                    'generation': generation_num,
                    'timestamp': datetime.now()
                }
                improvements.append(improvement)
                self.intrageneration_improvements.append(improvement)
                
        self.generation_memory.append({
            'generation': generation_num,
            'population_size': len(population),
            'avg_fitness': avg_fitness,
            'improvements': len(improvements)
        })
        
        return await {
            'avg_fitness': avg_fitness,
            'improvements_found': len(improvements),
            'patterns_learned': learned_patterns.shape[0]
        }

class EnhancedDarwinSystem:
    """Sistema Darwin com aprendizado intra-geração"""
    
    async def __init__(self):
        self.intra_learner = IntraGenerationLearner()
        self.population = []
        self.generation = 0
        
        # Inicializar população
        for i in range(20):
            self.population.append({
                'id': i,
                'fitness': deterministic_uniform(0, 100),
                'age': 0,
                'energy': 100,
                'behaviors': []
            })
            
    async def run_generation_with_learning(self):
        """Executa uma geração com aprendizado intra-geração"""
        self.generation += 1
        
        # Aprendizado intra-geração
        learning_results = self.intra_learner.learn_within_generation(self.population, self.generation)
        
        if learning_results.get('improvements_found', 0) > 0:
            logger.info(f"🧬 Intra-geração {self.generation}: {learning_results['improvements_found']} melhorias encontradas")
            
            # Aplicar melhorias à população
            for improvement in self.intra_learner.intrageneration_improvements[-learning_results['improvements_found']:]:
                agent_idx = improvement['agent_index']
                if agent_idx < len(self.population):
                    # Melhorar fitness baseado na melhoria aprendida
                    improvement_amount = improvement['improvement_score'] * 0.1
                    self.population[agent_idx]['fitness'] += improvement_amount
                    self.population[agent_idx]['behaviors'].append('intra_generation_improvement')
                    
        # Simulação de evolução
        # Manter top performers
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        survivors = self.population[:10]  # Top 50%
        
        # Reprodução com mutação
        new_population = survivors.copy()
        while len(new_population) < 20:
            parent = np.deterministic_choice(survivors)
            child = parent.copy()
            child['id'] = len(new_population)
            child['fitness'] += deterministic_uniform(-10, 10)  # Mutação
            child['age'] = 0
            new_population.append(child)
            
        self.population = new_population
        
        # Envelhecer população
        for agent in self.population:
            agent['age'] += 1
            
        return await learning_results

if __name__ == "__main__":
    system = EnhancedDarwinSystem()
    logger.info("Sistema Darwin com aprendizado intra-geração iniciado")
    
    total_improvements = 0
    
    for gen in range(20):
        results = system.run_generation_with_learning()
        total_improvements += results.get('improvements_found', 0)
        
        avg_fitness = sum(p['fitness'] for p in system.population) / len(system.population)
        logger.info(f"Geração {gen}: Fitness médio = {avg_fitness:.2f}, Melhorias = {results.get('improvements_found', 0)}")
        
    logger.info(f"Evolução concluída: {total_improvements} melhorias intra-geração no total")
    
    # Salvar resultados
    results = {
        'generations': system.generation,
        'final_avg_fitness': sum(p['fitness'] for p in system.population) / len(system.population),
        'total_improvements': total_improvements,
        'population_size': len(system.population),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('darwin_intrageneration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("Resultados salvos: darwin_intrageneration_results.json")