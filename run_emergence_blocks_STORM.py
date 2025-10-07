#!/usr/bin/env python3
"""
EMERGENCE BLOCKS - COM MUTATION STORM ATIVO
===========================================

Roda Darwin COM mutação 5× desde o INÍCIO!
"""

import logging
from pathlib import Path
from darwin_checkpoint_helper import save_darwin_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emergence_runner_storm")


def main():
    logger.info("=== EMERGENCE BLOCK RUNNER - MUTATION STORM ===")
    
    # FORÇA mutation storm
    import sys
    sys.path.insert(0, '/root/darwin-engine-intelligence')
    
    try:
        from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator
    except Exception as e:
        logger.error(f"Darwin orchestrator unavailable: {e}")
        return 1

    # Cria orchestrator COM storm
    orch = DarwinEvolutionOrchestrator()
    
    # OVERRIDE mutation rate para 1.0 (100%!)
    if hasattr(orch, 'mutation_rate'):
        orch.mutation_rate = 1.0
        logger.info("🔥 MUTATION STORM: mutation_rate = 1.0")
    
    # MNIST block COM storm
    logger.info("[BLOCK] MNIST: gens=100, pop=30, MUTATION STORM 🔥")
    best_mnist = orch.evolve_mnist(
        generations=100,  # 2× mais gerações
        population_size=30,  # 50% mais população
        demo_fast=True, 
        demo_epochs=6
    )
    logger.info(f"MNIST best fitness: {best_mnist.fitness:.4f}")
    try:
        gen = getattr(best_mnist, 'generation', 0)
        save_darwin_checkpoint(gen, best_mnist, None, task="mnist")
    except Exception as e:
        logger.warning(f"Falha ao salvar checkpoint MNIST: {e}")

    # CartPole block COM storm
    logger.info("[BLOCK] CARTPOLE: gens=100, pop=30, MUTATION STORM 🔥")
    best_cp = orch.evolve_cartpole(
        generations=100, 
        population_size=30, 
        demo_fast=True, 
        demo_epochs=3
    )
    logger.info(f"CartPole best fitness: {getattr(best_cp,'fitness',0.0):.4f}")
    try:
        gen = getattr(best_cp, 'generation', 100)
        ckpt_path = save_darwin_checkpoint(gen, best_cp, None, task="cartpole")
        if ckpt_path:
            logger.info(f"   Saved to: {ckpt_path}")
    except Exception as e:
        logger.error(f"❌ Falha ao salvar checkpoint CartPole: {e}")

    logger.info("=== DONE WITH STORM ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
