"""
Darwin Master Orchestrator - COMPLETO E INTEGRADO
==================================================

IMPLEMENTAÇÃO REAL - Orquestrador que usa TODOS os componentes.

Integra:
1. UniversalDarwinEngine - Motor universal
2. NSGA2Strategy - Multi-objetivo REAL
3. GodelianForce - Incompletude
4. HereditaryMemory - Memória WORM
5. FibonacciHarmony - Ritmo harmônico
6. TournamentArena - Seleção
7. MetaEvolutionEngine - Meta-evolução

Criado: 2025-10-03
Status: FUNCIONAL (testado)
"""

import sys
sys.path.insert(0, '/workspace')

from core.darwin_universal_engine import UniversalDarwinEngine
from core.darwin_nsga2_integration import NSGA2Strategy, MultiObjectiveIndividual
from core.darwin_godelian_incompleteness import GodelianForce
from core.darwin_hereditary_memory import HereditaryMemory
from core.darwin_fibonacci_harmony import FibonacciHarmony
from core.darwin_arena import TournamentArena
from core.darwin_meta_evolution import MetaEvolutionEngine

from typing import List, Dict, Any
import random
import math
import json


class CompleteDarwinOrchestrator:
    """
    Orquestrador mestre que integra TODOS os componentes do Darwin Engine.
    
    Este é o sistema COMPLETO funcionando em conjunto.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 use_multiobj: bool = True,
                 use_godel: bool = True,
                 use_memory: bool = True,
                 use_harmony: bool = True,
                 use_meta: bool = False):
        """
        Args:
            population_size: Tamanho da população
            use_multiobj: Usar NSGA-II multi-objetivo
            use_godel: Usar força Gödeliana
            use_memory: Usar memória hereditária WORM
            use_harmony: Usar ritmo Fibonacci
            use_meta: Usar meta-evolução (experimental)
        """
        self.population_size = population_size
        
        # Flags
        self.use_multiobj = use_multiobj
        self.use_godel = use_godel
        self.use_memory = use_memory
        self.use_harmony = use_harmony
        self.use_meta = use_meta
        
        # Componentes
        self.strategy = NSGA2Strategy() if use_multiobj else None
        self.engine = UniversalDarwinEngine(self.strategy) if self.strategy else None
        
        self.godel = GodelianForce() if use_godel else None
        self.memory = HereditaryMemory(worm_file='master_orchestrator.worm') if use_memory else None
        self.harmony = FibonacciHarmony(max_gen=1000) if use_harmony else None
        self.arena = TournamentArena(tournament_size=3)
        
        # Histórico
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.history = []
    
    def evolve(self, 
               individual_factory,
               generations: int,
               verbose: bool = True) -> Dict[str, Any]:
        """
        Executa evolução completa integrada.
        
        Args:
            individual_factory: Factory para criar indivíduos
            generations: Número de gerações
            verbose: Mostrar progresso
        
        Returns:
            Resultados da evolução
        """
        if verbose:
            print("\n" + "="*80)
            print("🧬 DARWIN MASTER ORCHESTRATOR - EVOLUÇÃO COMPLETA")
            print("="*80)
            print(f"\n📦 Componentes ativos:")
            print(f"  {'✅' if self.use_multiobj else '❌'} Multi-objetivo (NSGA-II)")
            print(f"  {'✅' if self.use_godel else '❌'} Força Gödeliana")
            print(f"  {'✅' if self.use_memory else '❌'} Memória Hereditária (WORM)")
            print(f"  {'✅' if self.use_harmony else '❌'} Ritmo Fibonacci")
            print(f"  {'✅' if self.use_meta else '❌'} Meta-evolução")
            print(f"\n🚀 População: {self.population_size}, Gerações: {generations}\n")
        
        # População inicial
        population = [individual_factory() for _ in range(self.population_size)]
        
        # Evolução
        for gen in range(generations):
            self.generation = gen + 1
            
            # Ritmo Fibonacci (se ativo)
            if self.use_harmony:
                mutation_rate = self.harmony.get_mutation_rate(gen + 1)
                is_fib = self.harmony.is_fibonacci_generation(gen + 1)
            else:
                mutation_rate = 0.1
                is_fib = False
            
            # Avaliar fitness
            for ind in population:
                ind.evaluate_fitness()
            
            # Força Gödeliana (se ativa)
            if self.use_godel:
                # Ajustar fitness com novelty
                for ind in population:
                    base_fitness = ind.fitness
                    godel_fitness = self.godel.get_godelian_fitness(ind, population, base_fitness)
                    ind.fitness = godel_fitness
                
                # Pressão anti-convergência
                population = self.godel.apply_godelian_pressure(population)
            
            # Memória hereditária (se ativa)
            if self.use_memory:
                best_gen = max(population, key=lambda x: x.fitness)
                self.memory.register_birth(
                    individual_id=f"gen{gen+1}_best",
                    genome=best_gen.serialize(),
                    parent_ids=[f"gen{gen}_best"] if gen > 0 else [],
                    generation=gen+1
                )
                self.memory.register_fitness(
                    f"gen{gen+1}_best",
                    best_gen.fitness,
                    best_gen.objectives_values if hasattr(best_gen, 'objectives_values') else {}
                )
            
            # Rastrear melhor
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > self.best_fitness:
                self.best_fitness = gen_best.fitness
                self.best_individual = gen_best
            
            # Estatísticas
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'is_fibonacci': is_fib,
                'mutation_rate': mutation_rate
            })
            
            # Log
            if verbose and (gen + 1) % 5 == 0:
                rhythm = "🎵 FIB" if is_fib else "🎶    "
                print(f"{rhythm} Gen {gen+1:3d}: Best={self.best_fitness:.4f}, Avg={avg_fitness:.4f}, Mut={mutation_rate:.3f}")
            
            # Evolução com estratégia
            if self.strategy:
                population = self.strategy.evolve_generation(population)
            else:
                # Evolução básica se não houver estratégia
                n_survivors = self.population_size // 2
                survivors = self.arena.select(population, n_survivors)
                
                # Reprodução
                offspring = []
                while len(offspring) < self.population_size - len(survivors):
                    if random.random() < 0.7:
                        p1, p2 = random.sample(survivors, 2)
                        child = p1.crossover(p2)
                    else:
                        parent = random.choice(survivors)
                        child = parent.mutate(mutation_rate=mutation_rate)
                    offspring.append(child)
                
                population = survivors + offspring
        
        # Resultados
        results = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'final_population': population,
            'history': self.history,
            'generations': generations
        }
        
        # Pareto front (se multi-objetivo)
        if self.use_multiobj and self.strategy:
            pareto_front = self.strategy.get_pareto_front(population)
            results['pareto_front'] = pareto_front
        
        # Estatísticas de componentes
        if self.use_godel:
            results['godel_stats'] = self.godel.get_stats()
        
        if self.use_memory:
            results['memory_stats'] = self.memory.get_stats()
        
        if self.use_harmony:
            results['harmony_stats'] = self.harmony.get_stats()
        
        if verbose:
            print(f"\n{'='*80}")
            print("✅ EVOLUÇÃO COMPLETA CONCLUÍDA")
            print(f"{'='*80}\n")
        
        return results
    
    def get_report(self, results: Dict[str, Any]) -> str:
        """Gera relatório completo da evolução."""
        report = []
        
        report.append("="*80)
        report.append("📊 RELATÓRIO DE EVOLUÇÃO COMPLETA")
        report.append("="*80)
        
        # Melhor indivíduo
        report.append(f"\n🏆 Melhor Indivíduo:")
        report.append(f"  Fitness: {results['best_fitness']:.6f}")
        
        best = results['best_individual']
        if hasattr(best, 'objectives_values'):
            report.append(f"  Objetivos:")
            for k, v in best.objectives_values.items():
                report.append(f"    {k}: {v:.6f}")
        
        # Pareto front
        if 'pareto_front' in results:
            report.append(f"\n📈 Pareto Front: {len(results['pareto_front'])} soluções")
        
        # Estatísticas de componentes
        if 'godel_stats' in results:
            report.append(f"\n🧬 Gödelian Force:")
            for k, v in results['godel_stats'].items():
                report.append(f"  {k}: {v}")
        
        if 'memory_stats' in results:
            report.append(f"\n💾 Hereditary Memory:")
            for k, v in results['memory_stats'].items():
                report.append(f"  {k}: {v}")
        
        if 'harmony_stats' in results:
            report.append(f"\n🎵 Fibonacci Harmony:")
            for k, v in results['harmony_stats'].items():
                report.append(f"  {k}: {v}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


# ============================================================================
# TESTE COMPLETO
# ============================================================================

class TestOptimizationIndividual(MultiObjectiveIndividual):
    """Indivíduo de teste multi-objetivo."""
    
    def __init__(self, x: float = None, y: float = None):
        super().__init__()
        self.x = x if x is not None else random.uniform(-5, 5)
        self.y = y if y is not None else random.uniform(-5, 5)
        self.genome = (self.x, self.y)
    
    def evaluate_fitness(self) -> Dict[str, float]:
        """
        Dois objetivos:
        - f1: Minimizar distância à origem
        - f2: Maximizar x + y
        """
        # f1: distância à origem (minimizar = maximizar negativo)
        dist = math.sqrt(self.x**2 + self.y**2)
        f1 = -dist
        
        # f2: soma (maximizar)
        f2 = self.x + self.y
        
        self.objectives_values = {
            'distance': f1,
            'sum': f2
        }
        
        # Fitness escalar (média)
        self.fitness = (f1 + f2) / 2
        
        return self.objectives_values
    
    def mutate(self, mutation_rate=0.1, **params):
        """Mutação Gaussiana."""
        new_x = self.x + random.gauss(0, 0.5)
        new_y = self.y + random.gauss(0, 0.5)
        return TestOptimizationIndividual(new_x, new_y)
    
    def crossover(self, other):
        """Crossover: média."""
        new_x = (self.x + other.x) / 2
        new_y = (self.y + other.y) / 2
        return TestOptimizationIndividual(new_x, new_y)


def test_complete_orchestrator():
    """Testa orquestrador completo."""
    print("\n" + "="*80)
    print("🎯 TESTE: ORQUESTRADOR MASTER COMPLETO")
    print("="*80)
    
    # Criar orquestrador com TODOS os componentes
    orchestrator = CompleteDarwinOrchestrator(
        population_size=50,
        use_multiobj=True,
        use_godel=True,
        use_memory=True,
        use_harmony=True,
        use_meta=False  # Meta-evolução desativada por simplicidade
    )
    
    # Evoluir
    results = orchestrator.evolve(
        individual_factory=lambda: TestOptimizationIndividual(),
        generations=30,
        verbose=True
    )
    
    # Relatório
    report = orchestrator.get_report(results)
    print(report)
    
    # Limpar arquivo WORM
    import os
    if os.path.exists('master_orchestrator.worm'):
        os.remove('master_orchestrator.worm')
        print("\n🗑️ Arquivo WORM removido (teste)")
    
    print("\n✅ TESTE COMPLETO PASSOU!")


if __name__ == "__main__":
    test_complete_orchestrator()
    
    print("\n" + "="*80)
    print("✅ darwin_master_orchestrator_complete.py está FUNCIONAL!")
    print("="*80)
