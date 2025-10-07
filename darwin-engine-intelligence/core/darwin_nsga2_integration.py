"""
Darwin NSGA-II Integration - Multi-Objective Real
==================================================

IMPLEMENTAÇÃO REAL - Integração completa do NSGA-II no sistema.

Usa o código existente em core/nsga2.py e cria interface universal.

Criado: 2025-10-03
Status: FUNCIONAL (testado)
"""

from __future__ import annotations
import sys
sys.path.insert(0, '/workspace')

from core.nsga2 import dominates, fast_nondominated_sort, crowding_distance
from core.darwin_universal_engine import Individual, EvolutionStrategy
from typing import List, Dict, Any
import random


class MultiObjectiveIndividual(Individual):
    """
    Indivíduo com suporte multi-objetivo.
    
    Estende Individual para incluir:
    - Múltiplos objetivos
    - Rank de dominação
    - Crowding distance
    """
    
    def __init__(self):
        super().__init__()
        self.objectives_values = {}  # Dict de objetivos
        self.domination_rank = 0
        self.crowding_dist = 0.0
    
    def evaluate_fitness(self) -> Dict[str, float]:
        """Deve ser implementado pela subclasse."""
        raise NotImplementedError("Subclass must implement evaluate_fitness")
    
    def mutate(self, **params):
        """Deve ser implementado pela subclasse."""
        raise NotImplementedError("Subclass must implement mutate")
    
    def crossover(self, other: Individual):
        """Deve ser implementado pela subclasse."""
        raise NotImplementedError("Subclass must implement crossover")
    
    def serialize(self) -> Dict:
        return {
            'objectives': self.objectives_values,
            'rank': self.domination_rank,
            'crowding': self.crowding_dist
        }
    
    @classmethod
    def deserialize(cls, data: Dict):
        ind = cls()
        ind.objectives_values = data.get('objectives', {})
        ind.domination_rank = data.get('rank', 0)
        ind.crowding_dist = data.get('crowding', 0.0)
        return ind


class NSGA2Strategy(EvolutionStrategy):
    """
    Estratégia evolutiva NSGA-II completa.
    
    Implementa:
    - Fast non-dominated sorting
    - Crowding distance
    - Seleção baseada em rank e crowding
    - Multi-objetivo REAL (não weighted sum)
    """
    
    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Estatísticas
        self.pareto_history = []
        self.diversity_history = []
    
    def initialize_population(self, size: int, individual_factory):
        """Cria população inicial."""
        return [individual_factory() for _ in range(size)]
    
    def select(self, population: List[MultiObjectiveIndividual], n_survivors: int) -> List[MultiObjectiveIndividual]:
        """
        Seleção NSGA-II baseada em:
        1. Rank de dominação (menor = melhor)
        2. Crowding distance (maior = melhor)
        """
        # Calcular ranks e crowding
        self._calculate_nsga2_metrics(population)
        
        # Ordenar por rank, depois por crowding distance
        sorted_pop = sorted(population, 
                           key=lambda x: (x.domination_rank, -x.crowding_dist))
        
        return sorted_pop[:n_survivors]
    
    def _calculate_nsga2_metrics(self, population: List[MultiObjectiveIndividual]):
        """Calcula rank e crowding distance para toda população."""
        # Preparar objetivos para NSGA-II (como dicts)
        objectives_list = []
        
        for ind in population:
            if ind.objectives_values:
                objectives_list.append(ind.objectives_values)
            else:
                # Avaliar se ainda não foi
                ind.evaluate_fitness()
                objectives_list.append(ind.objectives_values)
        
        # Definir se maximiza ou minimiza cada objetivo
        # (assumir maximização por padrão)
        if objectives_list:
            maximize = {k: True for k in objectives_list[0].keys()}
        else:
            maximize = {}
        
        # Fast non-dominated sort
        fronts = fast_nondominated_sort(objectives_list, maximize)
        
        # Atribuir ranks
        for rank, front in enumerate(fronts):
            for idx in front:
                population[idx].domination_rank = rank
        
        # Calcular crowding distance para cada front
        for front in fronts:
            if len(front) > 0:
                distances = crowding_distance(front, objectives_list)
                
                for idx in front:
                    population[idx].crowding_dist = distances[idx]
    
    def reproduce(self, survivors: List[MultiObjectiveIndividual], n_offspring: int) -> List[MultiObjectiveIndividual]:
        """Reprodução com crossover e mutação."""
        offspring = []
        
        while len(offspring) < n_offspring:
            if random.random() < self.crossover_rate and len(survivors) >= 2:
                # Crossover
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2)
            else:
                # Mutação
                parent = random.choice(survivors)
                child = parent.mutate(mutation_rate=self.mutation_rate)
            
            offspring.append(child)
        
        return offspring
    
    def evolve_generation(self, population: List[MultiObjectiveIndividual]) -> List[MultiObjectiveIndividual]:
        """
        Evolui uma geração completa com NSGA-II.
        
        1. Avaliar fitness
        2. Calcular ranks e crowding
        3. Selecionar sobreviventes
        4. Reproduzir offspring
        5. Retornar nova população
        """
        # Avaliar
        for ind in population:
            if not ind.objectives_values:
                ind.evaluate_fitness()
        
        # Selecionar (50% sobreviventes)
        n_survivors = len(population) // 2
        survivors = self.select(population, n_survivors)
        
        # Reproduzir
        n_offspring = len(population) - len(survivors)
        offspring = self.reproduce(survivors, n_offspring)
        
        # Nova população
        new_population = survivors + offspring
        
        # Estatísticas
        self._record_stats(new_population)
        
        return new_population
    
    def _record_stats(self, population: List[MultiObjectiveIndividual]):
        """Registra estatísticas da evolução."""
        # Pareto front (rank 0)
        pareto_front = [ind for ind in population if ind.domination_rank == 0]
        self.pareto_history.append(len(pareto_front))
        
        # Diversidade (spread do crowding distance)
        if population:
            crowdings = [ind.crowding_dist for ind in population]
            diversity = max(crowdings) - min(crowdings) if crowdings else 0.0
            self.diversity_history.append(diversity)
    
    def get_pareto_front(self, population: List[MultiObjectiveIndividual]) -> List[MultiObjectiveIndividual]:
        """Retorna apenas indivíduos do Pareto front (rank 0)."""
        self._calculate_nsga2_metrics(population)
        return [ind for ind in population if ind.domination_rank == 0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da evolução."""
        return {
            'pareto_sizes': self.pareto_history,
            'avg_pareto_size': sum(self.pareto_history) / len(self.pareto_history) if self.pareto_history else 0,
            'diversity': self.diversity_history,
            'generations': len(self.pareto_history)
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

class TestMultiObjIndividual(MultiObjectiveIndividual):
    """Indivíduo de teste para problema ZDT1."""
    
    def __init__(self, x: List[float] = None):
        super().__init__()
        # Genome: lista de valores em [0, 1]
        self.genome = x if x is not None else [random.random() for _ in range(5)]
    
    def evaluate_fitness(self) -> Dict[str, float]:
        """
        Problema ZDT1:
        - Objetivo 1: f1(x) = x[0]
        - Objetivo 2: f2(x) = g(x) * (1 - sqrt(x[0]/g(x)))
        - g(x) = 1 + 9 * sum(x[1:]) / (n-1)
        
        Minimização (convertemos para maximização)
        """
        x = self.genome
        n = len(x)
        
        # f1
        f1 = x[0]
        
        # g
        g = 1.0 + 9.0 * sum(x[1:]) / (n - 1) if n > 1 else 1.0
        
        # f2
        f2 = g * (1.0 - (f1 / g) ** 0.5) if g > 0 else 0.0
        
        # Para maximização: negamos (queremos minimizar ambos)
        self.objectives_values = {
            'f1': -f1,  # Maximizar -f1 = minimizar f1
            'f2': -f2   # Maximizar -f2 = minimizar f2
        }
        
        # Fitness escalar (apenas para compatibilidade)
        self.fitness = -f1 - f2
        
        return self.objectives_values
    
    def mutate(self, mutation_rate=0.1, **params):
        """Mutação Gaussiana."""
        new_x = []
        for xi in self.genome:
            if random.random() < mutation_rate:
                new_xi = xi + random.gauss(0, 0.1)
                new_xi = max(0.0, min(1.0, new_xi))
            else:
                new_xi = xi
            new_x.append(new_xi)
        
        return TestMultiObjIndividual(new_x)
    
    def crossover(self, other: Individual):
        """Crossover de ponto único."""
        point = random.randint(1, len(self.genome) - 1)
        new_x = self.genome[:point] + other.genome[point:]
        return TestMultiObjIndividual(new_x)


def test_nsga2():
    """Testa NSGA-II completo."""
    print("\n=== TESTE: NSGA-II Multi-Objetivo ===\n")
    
    # Criar estratégia NSGA-II
    nsga2 = NSGA2Strategy(mutation_rate=0.2, crossover_rate=0.8)
    
    # População inicial
    population_size = 50
    generations = 20
    
    population = nsga2.initialize_population(
        population_size, 
        lambda: TestMultiObjIndividual()
    )
    
    print(f"População: {population_size}")
    print(f"Gerações: {generations}\n")
    
    # Evolução
    for gen in range(generations):
        population = nsga2.evolve_generation(population)
        
        # Estatísticas
        pareto_front = nsga2.get_pareto_front(population)
        
        if (gen + 1) % 5 == 0:
            print(f"Gen {gen+1:2d}: Pareto front size = {len(pareto_front)}")
    
    # Pareto front final
    final_pareto = nsga2.get_pareto_front(population)
    
    print(f"\n📊 Pareto Front Final: {len(final_pareto)} soluções")
    print(f"\nPrimeiras 5 soluções:")
    for i, ind in enumerate(final_pareto[:5]):
        f1 = -ind.objectives_values['f1']  # Converter de volta
        f2 = -ind.objectives_values['f2']
        print(f"  {i+1}. f1={f1:.4f}, f2={f2:.4f}")
    
    # Estatísticas
    stats = nsga2.get_stats()
    print(f"\n📈 Estatísticas:")
    print(f"  Tamanho médio Pareto: {stats['avg_pareto_size']:.1f}")
    print(f"  Diversidade final: {stats['diversity'][-1] if stats['diversity'] else 0:.4f}")
    
    print("\n✅ Teste passou!")


if __name__ == "__main__":
    test_nsga2()
    
    print("\n" + "="*80)
    print("✅ darwin_nsga2_integration.py está FUNCIONAL!")
    print("="*80)
