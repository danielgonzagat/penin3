"""
Island Model: Distributed Evolutionary Algorithm
================================================

IMPLEMENTAÇÃO PURA PYTHON (SEM DEPENDENCIES EXTERNAS)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

Modelo de ilhas permite evolução paralela com migração periódica,
aumentando diversidade e escalabilidade.

Based on: Whitley et al. (1999) "Island Model Genetic Algorithms"
"""

import random
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum


class MigrationTopology(Enum):
    """Topologias de migração entre ilhas"""
    RING = "ring"  # Cada ilha → próxima (circular)
    FULLY_CONNECTED = "fully_connected"  # Todas ↔ todas
    STAR = "star"  # Hub central ↔ todas periféricas
    RANDOM = "random"  # Conexões aleatórias


@dataclass
class Island:
    """Uma ilha evolucionária"""
    island_id: int
    population: List[Any]
    best_individual: Any = None
    best_fitness: float = float('inf')
    generations: int = 0
    immigrants_received: int = 0
    emigrants_sent: int = 0


class IslandModel:
    """
    Modelo de Ilhas para evolução distribuída
    
    Características:
    - Múltiplas populações isoladas (ilhas)
    - Migração periódica entre ilhas
    - Topologias configuráveis
    - Elitismo por ilha
    - Escalável
    """
    
    def __init__(self,
                 n_islands: int,
                 population_size_per_island: int,
                 individual_factory: Callable,
                 fitness_fn: Callable[[Any], float],
                 mutation_fn: Callable[[Any], Any],
                 crossover_fn: Callable[[Any, Any], Any],
                 topology: MigrationTopology = MigrationTopology.RING,
                 migration_rate: float = 0.1,
                 migration_interval: int = 10):
        """
        Args:
            n_islands: Número de ilhas
            population_size_per_island: População por ilha
            individual_factory: Cria novo indivíduo
            fitness_fn: Avalia fitness (minimizar)
            mutation_fn: Mutaciona indivíduo
            crossover_fn: Crossover entre 2 indivíduos
            topology: Topologia de migração
            migration_rate: % da população que migra
            migration_interval: Gerações entre migrações
        """
        self.n_islands = n_islands
        self.pop_size = population_size_per_island
        self.individual_factory = individual_factory
        self.fitness_fn = fitness_fn
        self.mutation_fn = mutation_fn
        self.crossover_fn = crossover_fn
        self.topology = topology
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        
        # Criar ilhas
        self.islands: List[Island] = []
        for i in range(n_islands):
            population = [individual_factory() for _ in range(population_size_per_island)]
            island = Island(island_id=i, population=population)
            self.islands.append(island)
        
        # Estado global
        self.global_generation = 0
        self.global_best_fitness = float('inf')
        self.global_best_individual = None
        self.migration_history = []
        
        print(f"\n🏝️ Island Model inicializado:")
        print(f"   Ilhas: {n_islands}")
        print(f"   Pop/ilha: {population_size_per_island}")
        print(f"   Pop total: {n_islands * population_size_per_island}")
        print(f"   Topologia: {topology.value}")
        print(f"   Migração: {migration_rate*100:.0f}% cada {migration_interval} gens\n")
    
    def _evaluate_island(self, island: Island):
        """Avalia todos indivíduos de uma ilha"""
        for ind in island.population:
            if not hasattr(ind, 'fitness'):
                ind.fitness = self.fitness_fn(ind)
            elif ind.fitness is None:
                ind.fitness = self.fitness_fn(ind)
        
        # Atualizar melhor da ilha
        best = min(island.population, key=lambda x: x.fitness)
        if best.fitness < island.best_fitness:
            island.best_fitness = best.fitness
            island.best_individual = best
        
        # Atualizar melhor global
        if best.fitness < self.global_best_fitness:
            self.global_best_fitness = best.fitness
            self.global_best_individual = best
    
    def _evolve_island(self, island: Island):
        """Evolui uma ilha por uma geração"""
        # Seleção por torneio
        def tournament_select(k=3):
            candidates = random.sample(island.population, min(k, len(island.population)))
            return min(candidates, key=lambda x: x.fitness)
        
        # Elite
        sorted_pop = sorted(island.population, key=lambda x: x.fitness)
        elite_size = max(1, int(self.pop_size * 0.1))
        new_population = sorted_pop[:elite_size]
        
        # Gerar offspring
        while len(new_population) < self.pop_size:
            if random.random() < 0.8:  # Crossover
                p1 = tournament_select()
                p2 = tournament_select()
                child = self.crossover_fn(p1, p2)
            else:  # Mutação pura
                parent = tournament_select()
                child = self.mutation_fn(parent)
            
            # Mutar offspring
            if random.random() < 0.2:
                child = self.mutation_fn(child)
            
            new_population.append(child)
        
        island.population = new_population[:self.pop_size]
        island.generations += 1
    
    def _get_migration_targets(self, source_island_id: int) -> List[int]:
        """Determina ilhas alvo para migração"""
        if self.topology == MigrationTopology.RING:
            # Próxima ilha no anel
            return [(source_island_id + 1) % self.n_islands]
        
        elif self.topology == MigrationTopology.FULLY_CONNECTED:
            # Todas as outras ilhas
            return [i for i in range(self.n_islands) if i != source_island_id]
        
        elif self.topology == MigrationTopology.STAR:
            # Hub = ilha 0
            if source_island_id == 0:
                return list(range(1, self.n_islands))
            else:
                return [0]
        
        elif self.topology == MigrationTopology.RANDOM:
            # 1-2 ilhas aleatórias
            n_targets = random.randint(1, 2)
            candidates = [i for i in range(self.n_islands) if i != source_island_id]
            return random.sample(candidates, min(n_targets, len(candidates)))
        
        return []
    
    def _migrate(self):
        """Realiza migração entre ilhas"""
        # Preparar migrantes de cada ilha
        migrants_per_island = {}
        
        for island in self.islands:
            # Garantir que todos têm fitness
            for ind in island.population:
                if not hasattr(ind, 'fitness') or ind.fitness is None:
                    ind.fitness = self.fitness_fn(ind)
            
            # Selecionar melhores para migração
            n_migrants = max(1, int(self.pop_size * self.migration_rate))
            sorted_pop = sorted(island.population, key=lambda x: x.fitness)
            migrants = sorted_pop[:n_migrants]
            
            migrants_per_island[island.island_id] = migrants
            island.emigrants_sent += len(migrants)
        
        # Enviar migrantes para ilhas alvo
        for source_id, migrants in migrants_per_island.items():
            targets = self._get_migration_targets(source_id)
            
            for target_id in targets:
                if migrants:
                    # Enviar um migrante para cada alvo
                    migrant = random.choice(migrants)
                    target_island = self.islands[target_id]
                    
                    # Substituir pior indivíduo da ilha alvo
                    worst_idx = max(range(len(target_island.population)),
                                   key=lambda i: target_island.population[i].fitness)
                    target_island.population[worst_idx] = migrant
                    target_island.immigrants_received += 1
                    
                    # Log migração
                    self.migration_history.append({
                        'generation': self.global_generation,
                        'source': source_id,
                        'target': target_id,
                        'fitness': migrant.fitness
                    })
    
    def evolve(self, n_generations: int, verbose: bool = True):
        """
        Evolui todas as ilhas
        
        Args:
            n_generations: Número de gerações
            verbose: Mostrar progresso
        """
        if verbose:
            print(f"🚀 Evolução Island Model iniciada ({n_generations} gens)\n")
        
        for gen in range(n_generations):
            self.global_generation = gen + 1
            
            # Avaliar todas ilhas
            for island in self.islands:
                self._evaluate_island(island)
            
            # Evolve cada ilha
            for island in self.islands:
                self._evolve_island(island)
            
            # Migração periódica
            if (gen + 1) % self.migration_interval == 0:
                self._migrate()
                if verbose:
                    n_migrations = sum(1 for m in self.migration_history 
                                     if m['generation'] == self.global_generation)
                    print(f"   🔄 Migração: {n_migrations} migrantes trocados")
            
            # Log progress
            if verbose and (gen + 1) % 10 == 0:
                avg_best = sum(i.best_fitness for i in self.islands) / len(self.islands)
                print(f"Gen {gen+1:3d}: Global Best={self.global_best_fitness:.6e}, "
                      f"Avg Island Best={avg_best:.6e}")
        
        if verbose:
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Imprime estatísticas finais"""
        print(f"\n{'='*80}")
        print("📊 ESTATÍSTICAS FINAIS ISLAND MODEL")
        print(f"{'='*80}")
        print(f"  Global Best Fitness: {self.global_best_fitness:.6e}")
        print(f"  Total Gerações: {self.global_generation}")
        print(f"  Total Migrações: {len(self.migration_history)}")
        
        print(f"\n  📈 Estatísticas por Ilha:")
        for island in self.islands:
            print(f"     Ilha {island.island_id}: Best={island.best_fitness:.6e}, "
                  f"Gens={island.generations}, "
                  f"↑{island.immigrants_received}/↓{island.emigrants_sent}")
        
        # Diversidade entre ilhas
        best_fitnesses = [i.best_fitness for i in self.islands]
        diversity = max(best_fitnesses) - min(best_fitnesses)
        print(f"\n  🌈 Diversidade (range best fitness): {diversity:.6e}")
        
        print(f"\n{'='*80}")


# ============================================================================
# TESTES
# ============================================================================

class SimpleIndividual:
    """Indivíduo simples para testes"""
    def __init__(self, genome=None):
        if genome is None:
            self.genome = [random.uniform(-5, 5) for _ in range(3)]
        else:
            self.genome = list(genome)
        self.fitness = None
    
    def __repr__(self):
        return f"Ind({self.genome[:2]}...)"


def test_island_model():
    """Testa Island Model"""
    print("\n" + "="*80)
    print("TESTE: Island Model")
    print("="*80 + "\n")
    
    # Função de fitness: Sphere
    def sphere_fitness(ind):
        return sum(x**2 for x in ind.genome)
    
    # Mutação
    def mutate(ind):
        new_genome = [x + random.gauss(0, 0.5) for x in ind.genome]
        return SimpleIndividual(new_genome)
    
    # Crossover
    def crossover(ind1, ind2):
        child_genome = [(g1 + g2) / 2 for g1, g2 in zip(ind1.genome, ind2.genome)]
        return SimpleIndividual(child_genome)
    
    # Criar Island Model
    island_model = IslandModel(
        n_islands=4,
        population_size_per_island=20,
        individual_factory=lambda: SimpleIndividual(),
        fitness_fn=sphere_fitness,
        mutation_fn=mutate,
        crossover_fn=crossover,
        topology=MigrationTopology.RING,
        migration_rate=0.1,
        migration_interval=5
    )
    
    # Evoluir
    island_model.evolve(n_generations=30, verbose=True)
    
    # Validar
    assert island_model.global_best_fitness < 1.0, f"Fitness muito alto: {island_model.global_best_fitness}"
    assert len(island_model.migration_history) > 0, "Deve ter migrações"
    
    print(f"\n✅ Teste Island Model PASSOU!")
    print(f"   Best fitness: {island_model.global_best_fitness:.6e}")
    print(f"   Migrações: {len(island_model.migration_history)}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    random.seed(42)
    test_island_model()
    print("\n✅ island_model_pure.py FUNCIONAL!")
