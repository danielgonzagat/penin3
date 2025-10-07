#!/usr/bin/env python3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emergence_runner")


def main():
    logger.info("=== EMERGENCE BLOCK RUNNER ===")
    try:
        from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator
    except Exception as e:
        logger.error(f"Darwin orchestrator unavailable: {e}")
        return 1

    orch = DarwinEvolutionOrchestrator()

    # MNIST block
    logger.info("[BLOCK] MNIST: gens=50, pop=20, demo_fast=True")
    best_mnist = orch.evolve_mnist(generations=50, population_size=20, demo_fast=True, demo_epochs=6)
    logger.info(f"MNIST best fitness: {best_mnist.fitness:.4f} | genome={best_mnist.genome}")

    # CartPole block
    logger.info("[BLOCK] CARTPOLE: gens=50, pop=20, demo_fast=True")
    best_cp = orch.evolve_cartpole(generations=50, population_size=20, demo_fast=True, demo_epochs=3)
    logger.info(f"CartPole best fitness: {getattr(best_cp,'fitness',0.0):.4f} | genome={getattr(best_cp,'genome',{})}")

    logger.info("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
