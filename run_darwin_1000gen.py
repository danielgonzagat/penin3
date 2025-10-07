#!/usr/bin/env python3
"""
Darwin 1000 Gerações Isolado
=============================

Executa Darwin por 1000 gerações PURAS sem outras cargas.
Objetivo: Observar se inteligência emerge com evolução prolongada.
"""

import sys
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/darwin_1000gen.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("darwin_1000gen")

# Import Darwin
sys.path.insert(0, '/root/darwin-engine-intelligence')
from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator

def main():
    logger.info("="*80)
    logger.info("🧬 DARWIN 1000 GERAÇÕES - ISOLADO E FOCADO")
    logger.info("="*80)
    logger.info("")
    logger.info("Objetivo: Evolução pura por 1000 gerações para observar emergência")
    logger.info("População: 50")
    logger.info("Task: MNIST classification")
    logger.info("Checkpoints: A cada 100 gerações")
    logger.info("")
    
    start_time = time.time()
    
    try:
        # Criar orchestrator
        logger.info("Inicializando Darwin Orchestrator...")
        orch = DarwinEvolutionOrchestrator()
        
        # Configurar para evolução prolongada
        logger.info("\nConfigurando para evolução prolongada...")
        logger.info(f"   Mutation rate: {orch.mutation_rate}")
        logger.info(f"   Crossover probability: {orch.crossover_prob}")
        logger.info("")
        
        # Executar evolução de 1000 gerações
        logger.info("🚀 Iniciando evolução de 1000 gerações...\n")
        
        best_individual = orch.evolve_mnist(
            generations=1000,
            population_size=50,
            demo_fast=False,  # Treino completo!
            demo_epochs=10,    # Mais épocas por geração
            checkpoint_every=100  # Checkpoint a cada 100 gerações
        )
        
        elapsed = time.time() - start_time
        
        # Resultados finais
        logger.info("\n" + "="*80)
        logger.info("✅ EVOLUÇÃO DE 1000 GERAÇÕES COMPLETA!")
        logger.info("="*80)
        logger.info(f"   Tempo total: {elapsed/3600:.2f} horas")
        logger.info(f"   Best fitness: {best_individual.fitness:.4f}")
        
        if hasattr(best_individual, 'genome'):
            logger.info(f"   Best genome: {best_individual.genome}")
        
        # Salvar checkpoint final
        from darwin_checkpoint_helper import save_darwin_checkpoint
        checkpoint_path = save_darwin_checkpoint(
            1000, 
            best_individual, 
            None, 
            task="mnist_1000gen"
        )
        
        if checkpoint_path:
            logger.info(f"   Checkpoint final: {checkpoint_path}")
        
        logger.info("\n🎯 Análise: Verificar se fitness >0.95 indica emergência")
        logger.info("="*80 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrompido pelo usuário")
        elapsed = time.time() - start_time
        logger.info(f"   Tempo decorrido: {elapsed/3600:.2f} horas")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Erro durante evolução: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
