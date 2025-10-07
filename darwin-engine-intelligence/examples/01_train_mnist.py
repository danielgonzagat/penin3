"""
Exemplo 1: Treinar MNIST com Darwin Engine
===========================================

Demonstra como usar evolução darwiniana para otimizar
um classificador MNIST.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator

def main():
    print("="*80)
    print("EXEMPLO 1: Evoluindo Classificador MNIST")
    print("="*80)
    
    # Criar orquestrador
    orchestrator = DarwinEvolutionOrchestrator()
    
    # Evoluir MNIST
    print("\nIniciando evolução...")
    print("População: 100 indivíduos")
    print("Gerações: 100")
    print("Tempo estimado: 3-4 horas\n")
    
    best_individual = orchestrator.evolve_mnist(
        generations=100,
        population_size=100
    )
    
    print("\n" + "="*80)
    print("RESULTADO FINAL")
    print("="*80)
    print(f"Best fitness: {best_individual.fitness:.4f}")
    print(f"Best genome: {best_individual.genome}")
    print("\nEsperado: fitness 0.95-0.97 (95-97% accuracy)")

if __name__ == "__main__":
    main()
