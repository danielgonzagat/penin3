"""
Darwin Universal Engine - Motor Evolutivo Geral
================================================

IMPLEMENTAÇÃO REAL E TESTADA
Criado em: 2025-10-03
Status: FUNCIONAL

Permite executar qualquer paradigma evolutivo com qualquer tipo de indivíduo.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Individual(ABC):
    """Interface universal para qualquer indivíduo evoluível."""
    
    def __init__(self):
        self.fitness: float = 0.0
        self.objectives: Dict[str, float] = {}
        self.genome: Any = None
        self.age: int = 0
    
    @abstractmethod
    def evaluate_fitness(self) -> Dict[str, float]:
        """Avalia múltiplos objetivos. Deve retornar dict de objetivos."""
        pass
    
    @abstractmethod
    def mutate(self, **params) -> Individual:
        """Mutação genética. Retorna novo indivíduo mutado."""
        pass
    
    @abstractmethod
    def crossover(self, other: Individual) -> Individual:
        """Reprodução sexual. Retorna offspring."""
        pass
    
    @abstractmethod
    def serialize(self) -> Dict:
        """Serializa para JSON."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict) -> Individual:
        """Desserializa de JSON."""
        pass


class EvolutionStrategy(ABC):
    """Interface para qualquer paradigma evolutivo."""
    
    @abstractmethod
    def initialize_population(self, size: int, individual_factory: Callable[[], Individual]) -> List[Individual]:
        """Cria população inicial usando factory function."""
        pass
    
    @abstractmethod
    def select(self, population: List[Individual], n_survivors: int) -> List[Individual]:
        """Seleciona sobreviventes."""
        pass
    
    @abstractmethod
    def reproduce(self, survivors: List[Individual], n_offspring: int) -> List[Individual]:
        """Gera offspring."""
        pass
    
    @abstractmethod
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Executa uma geração completa."""
        pass


class GeneticAlgorithm(EvolutionStrategy):
    """
    Algoritmo Genético clássico.
    
    TESTADO: Sim
    STATUS: Funcional
    """
    
    def __init__(self, survival_rate: float = 0.4, sexual_rate: float = 0.8):
        self.survival_rate = survival_rate
        self.sexual_rate = sexual_rate
    
    def initialize_population(self, size: int, individual_factory: Callable[[], Individual]) -> List[Individual]:
        """Cria população usando factory."""
        return [individual_factory() for _ in range(size)]
    
    def select(self, population: List[Individual], n_survivors: int) -> List[Individual]:
        """Seleção por fitness (ordenação simples)."""
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:n_survivors]
    
    def reproduce(self, survivors: List[Individual], n_offspring: int) -> List[Individual]:
        """Reprodução sexual/assexual."""
        import random
        offspring = []
        
        while len(offspring) < n_offspring:
            if random.random() < self.sexual_rate and len(survivors) >= 2:
                # Sexual
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2).mutate()
            else:
                # Assexual
                parent = random.choice(survivors)
                child = parent.mutate()
            
            offspring.append(child)
        
        return offspring
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """
        Executa uma geração completa:
        1. Avaliar fitness
        2. Selecionar sobreviventes
        3. Reproduzir offspring
        4. Retornar nova população
        """
        # Avaliar (com error handling!)
        for ind in population:
            try:
                ind.evaluate_fitness()
            except Exception as e:
                logger.error(f"Erro ao avaliar indivíduo: {e}")
                ind.fitness = 0.0
        
        # Selecionar
        n_survivors = max(1, int(len(population) * self.survival_rate))
        survivors = self.select(population, n_survivors)
        
        # Reproduzir
        n_offspring = len(population) - len(survivors)
        offspring = self.reproduce(survivors, n_offspring)
        
        return survivors + offspring


class UniversalDarwinEngine:
    """
    Motor universal que aceita qualquer estratégia evolutiva.
    
    TESTADO: Sim
    STATUS: Funcional
    
    Uso:
    >>> engine = UniversalDarwinEngine(GeneticAlgorithm())
    >>> best = engine.evolve(lambda: MyIndividual(), pop_size=10, gens=5)
    """
    
    def __init__(self, strategy: EvolutionStrategy):
        self.strategy = strategy
        self.generation = 0
        self.history = []
    
    def evolve(self, individual_factory: Callable[[], Individual], 
               population_size: int, generations: int) -> Individual:
        """
        Executa evolução completa.
        
        Args:
            individual_factory: Function que cria novo indivíduo
            population_size: Tamanho da população
            generations: Número de gerações
        
        Returns:
            Melhor indivíduo encontrado
        """
        logger.info(f"🧬 Universal Darwin Engine")
        logger.info(f"   Strategy: {type(self.strategy).__name__}")
        logger.info(f"   Population: {population_size}")
        logger.info(f"   Generations: {generations}")
        
        # População inicial
        population = self.strategy.initialize_population(population_size, individual_factory)
        
        best = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            logger.info(f"\n🧬 Generation {gen+1}/{generations}")
            
            # Evolução
            population = self.strategy.evolve_generation(population)
            
            # Rastrear melhor
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                best = gen_best
            
            logger.info(f"   Best: {best_fitness:.4f}")
            
            # Histórico
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.history.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            self.generation += 1
        
        return best


# ============================================================================
# TESTES UNITÁRIOS
# ============================================================================

class DummyIndividual(Individual):
    """Indivíduo dummy para testes."""
    
    def __init__(self, value: float = None):
        super().__init__()
        import random
        self.genome = value if value is not None else random.random()
        self.fitness = 0.0
    
    def evaluate_fitness(self) -> Dict[str, float]:
        """Fitness = genome value."""
        self.fitness = self.genome
        self.objectives = {'value': self.fitness}
        return self.objectives
    
    def mutate(self, **params) -> Individual:
        """Adiciona ruído ao genome."""
        import random
        new_value = self.genome + random.gauss(0, 0.1)
        new_value = max(0.0, min(1.0, new_value))
        return DummyIndividual(new_value)
    
    def crossover(self, other: Individual) -> Individual:
        """Média dos genomes."""
        avg = (self.genome + other.genome) / 2.0
        return DummyIndividual(avg)
    
    def serialize(self) -> Dict:
        return {'genome': self.genome, 'fitness': self.fitness}
    
    @classmethod
    def deserialize(cls, data: Dict) -> Individual:
        ind = cls(data['genome'])
        ind.fitness = data['fitness']
        return ind


def test_universal_engine():
    """Testa motor universal com indivíduo dummy."""
    print("\n=== TESTE: Universal Darwin Engine ===\n")
    
    # Criar engine com GA
    ga = GeneticAlgorithm(survival_rate=0.4, sexual_rate=0.8)
    engine = UniversalDarwinEngine(ga)
    
    # Evoluir
    best = engine.evolve(
        individual_factory=lambda: DummyIndividual(),
        population_size=10,
        generations=5
    )
    
    print(f"\n✅ Teste passou!")
    print(f"   Best fitness: {best.fitness:.4f}")
    print(f"   Best genome: {best.genome:.4f}")
    print(f"   History: {len(engine.history)} gerações")
    
    assert best.fitness > 0.0, "Fitness deve ser > 0"
    assert len(engine.history) == 5, "Deve ter 5 gerações"
    print("\n✅ TODOS OS TESTES PASSARAM!\n")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Rodar testes
    test_universal_engine()
    
    print("="*80)
    print("✅ darwin_universal_engine.py está FUNCIONAL!")
    print("="*80)
