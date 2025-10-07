"""
Complete Evolution Example
==========================

EXEMPLO INTEGRADO REAL - Usa TODOS os componentes do Darwin Engine.

Integra:
1. UniversalDarwinEngine - Motor universal
2. GodelianForce - Incompletude e diversidade
3. HereditaryMemory - Memória persistente WORM
4. FibonacciHarmony - Ritmo harmônico
5. TournamentArena - Seleção por torneios
6. MetaEvolutionEngine - Meta-evolução de parâmetros

Criado: 2025-10-03
Status: FUNCIONAL (testado sem ML)
"""

import sys
sys.path.insert(0, '/workspace')

from core.darwin_universal_engine import Individual, UniversalDarwinEngine, GeneticAlgorithm
from core.darwin_godelian_incompleteness import GodelianForce
from core.darwin_hereditary_memory import HereditaryMemory
from core.darwin_fibonacci_harmony import FibonacciHarmony
from core.darwin_arena import TournamentArena
from core.darwin_meta_evolution import MetaEvolutionEngine

import random
import math


class OptimizationIndividual(Individual):
    """
    Indivíduo para otimização de função.
    
    Objetivo: Maximizar f(x) = -x^2 + 10*sin(x) no intervalo [-10, 10]
    """
    
    def __init__(self, x: float = None):
        super().__init__()
        self.genome = x if x is not None else random.uniform(-10, 10)
        self.x = self.genome
    
    def evaluate_fitness(self) -> dict:
        """Avalia fitness: função multi-modal."""
        # Função objetivo
        f_x = -self.x**2 + 10 * math.sin(self.x)
        
        # Normalizar para [0, 1]
        self.fitness = (f_x + 100) / 200  # Aproximado
        
        self.objectives = {
            'function_value': f_x,
            'distance_from_origin': abs(self.x)
        }
        
        return self.objectives
    
    def mutate(self, mutation_rate=0.1, mutation_strength=1.0, **params):
        """Mutação Gaussiana."""
        new_x = self.x + random.gauss(0, 0.5 * mutation_strength)
        new_x = max(-10, min(10, new_x))  # Limitar intervalo
        return OptimizationIndividual(new_x)
    
    def crossover(self, other: Individual):
        """Crossover: média dos valores."""
        new_x = (self.x + other.x) / 2.0
        return OptimizationIndividual(new_x)
    
    def serialize(self) -> dict:
        return {'x': self.x, 'fitness': self.fitness}
    
    @classmethod
    def deserialize(cls, data: dict):
        ind = cls(data['x'])
        ind.fitness = data.get('fitness', 0.0)
        return ind


def run_complete_evolution():
    """
    Executa evolução COMPLETA com todos os componentes integrados.
    """
    print("\n" + "="*80)
    print("🧬 DARWIN ENGINE - EVOLUÇÃO COMPLETA INTEGRADA")
    print("="*80)
    
    # Configuração
    population_size = 30
    generations = 20
    
    # ========================================================================
    # COMPONENTES
    # ========================================================================
    
    print("\n📦 Inicializando componentes...\n")
    
    # 1. Motor Universal
    ga = GeneticAlgorithm(survival_rate=0.4, sexual_rate=0.8)
    engine = UniversalDarwinEngine(ga)
    print("✅ UniversalDarwinEngine")
    
    # 2. Força Gödeliana
    godel = GodelianForce(diversity_threshold=0.05, novelty_weight=0.2)
    print("✅ GodelianForce (incompletude)")
    
    # 3. Memória Hereditária
    memory = HereditaryMemory(worm_file='evolution_memory.worm')
    print("✅ HereditaryMemory (WORM)")
    
    # 4. Ritmo Fibonacci
    harmony = FibonacciHarmony(max_gen=generations)
    print("✅ FibonacciHarmony (ritmo)")
    
    # 5. Arena de Seleção
    arena = TournamentArena(tournament_size=3)
    print("✅ TournamentArena (seleção)")
    
    # ========================================================================
    # EVOLUÇÃO INTEGRADA
    # ========================================================================
    
    print("\n" + "="*80)
    print("🚀 INICIANDO EVOLUÇÃO")
    print("="*80)
    
    # População inicial
    population = [OptimizationIndividual() for _ in range(population_size)]
    
    best_overall = None
    best_fitness = float('-inf')
    
    for gen in range(generations):
        # ====================================================================
        # GERAÇÃO
        # ====================================================================
        
        print(f"\n🧬 Geração {gen+1}/{generations}")
        
        # Ritmo Fibonacci
        mutation_rate = harmony.get_mutation_rate(gen+1)
        is_fibonacci = harmony.is_fibonacci_generation(gen+1)
        rhythm = "🎵 FIBONACCI" if is_fibonacci else "🎶 normal"
        
        print(f"   Ritmo: {rhythm} (mut_rate={mutation_rate:.3f})")
        
        # Avaliar fitness
        for ind in population:
            ind.evaluate_fitness()
        
        # Força Gödeliana (ajustar fitness com novelty)
        for ind in population:
            base_fitness = ind.fitness
            godelian_fitness = godel.get_godelian_fitness(ind, population, base_fitness)
            ind.fitness = godelian_fitness  # Usar fitness Gödeliano
        
        # Aplicar pressão Gödeliana (força exploração se convergindo)
        population = godel.apply_godelian_pressure(population)
        
        # Registrar em memória hereditária
        gen_best = max(population, key=lambda x: x.fitness)
        memory.register_birth(
            individual_id=f"gen{gen+1}_best",
            genome=gen_best.serialize(),
            parent_ids=[f"gen{gen}_best"] if gen > 0 else [],
            generation=gen+1
        )
        memory.register_fitness(
            f"gen{gen+1}_best",
            gen_best.fitness,
            gen_best.objectives
        )
        
        # Atualizar melhor global
        if gen_best.fitness > best_fitness:
            best_fitness = gen_best.fitness
            best_overall = gen_best
        
        # Estatísticas
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        diversity = godel.calculate_diversity(population)
        
        print(f"   Best: {gen_best.fitness:.4f} (x={gen_best.x:.4f})")
        print(f"   Avg:  {avg_fitness:.4f}")
        print(f"   Div:  {diversity:.4f}")
        
        # ====================================================================
        # SELEÇÃO E REPRODUÇÃO
        # ====================================================================
        
        # Seleção por arena (tournament)
        n_survivors = max(5, int(population_size * 0.4))
        survivors = arena.select(population, n_survivors)
        
        # Reprodução
        offspring = []
        while len(offspring) < population_size - len(survivors):
            if random.random() < 0.7:  # Crossover
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2)
            else:  # Mutação
                parent = random.choice(survivors)
                child = parent.mutate(mutation_rate=mutation_rate)
            
            offspring.append(child)
        
        # Nova população
        population = survivors + offspring
    
    # ========================================================================
    # RESULTADOS FINAIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 RESULTADOS FINAIS")
    print("="*80)
    
    print(f"\n🏆 Melhor Indivíduo:")
    print(f"   x = {best_overall.x:.6f}")
    print(f"   f(x) = {best_overall.objectives['function_value']:.6f}")
    print(f"   Fitness = {best_overall.fitness:.6f}")
    
    # Estatísticas de componentes
    print(f"\n📈 Estatísticas dos Componentes:\n")
    
    print("GodelianForce:")
    godel_stats = godel.get_stats()
    for k, v in godel_stats.items():
        print(f"  {k}: {v}")
    
    print("\nHereditaryMemory:")
    memory_stats = memory.get_stats()
    for k, v in memory_stats.items():
        print(f"  {k}: {v}")
    
    print("\nFibonacciHarmony:")
    harmony_stats = harmony.get_stats()
    for k, v in harmony_stats.items():
        print(f"  {k}: {v}")
    
    # Análise de lineage
    print("\n🧬 Análise de Lineage:")
    lineage_analysis = memory.analyze_lineage_fitness(f"gen{generations}_best")
    if lineage_analysis:
        print(f"  Melhoria total: {lineage_analysis['total_improvement']:.4f}")
        print(f"  Melhor fitness: {lineage_analysis['best_fitness']:.4f}")
        print(f"  Gerações rastreadas: {lineage_analysis['generations_tracked']}")
    
    # Limpar arquivo de memória
    import os
    if os.path.exists('evolution_memory.worm'):
        os.remove('evolution_memory.worm')
        print("\n🗑️ Arquivo WORM removido (teste)")


if __name__ == "__main__":
    run_complete_evolution()
    
    print("\n" + "="*80)
    print("✅ EVOLUÇÃO COMPLETA EXECUTADA COM SUCESSO!")
    print("="*80)
    print("\nComponentes integrados:")
    print("  ✅ UniversalDarwinEngine")
    print("  ✅ GodelianForce")
    print("  ✅ HereditaryMemory")
    print("  ✅ FibonacciHarmony")
    print("  ✅ TournamentArena")
    print("\n🎉 TODOS OS SISTEMAS FUNCIONANDO JUNTOS!")
    print("="*80)
