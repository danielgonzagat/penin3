#!/usr/bin/env python3
"""
Example script demonstrating how to use the Fibonacci Engine.

This example shows:
1. Creating a custom configuration
2. Selecting an adapter
3. Running the engine
4. Accessing results
5. Saving outputs
"""

import numpy as np
from fibonacci_engine import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters import RLSyntheticAdapter

def main():
    print("="*60)
    print("Fibonacci Engine - Example Usage")
    print("="*60)
    print()
    
    # 1. Create configuration
    config = FibonacciConfig(
        max_generations=30,
        fib_depth=10,
        population=32,
        elites_grid=(10, 10),
        rollback_delta=0.02,
        seed=42,
        verbose=1,
    )
    
    print("✓ Configuration created")
    print(f"  - Generations: {config.max_generations}")
    print(f"  - Population: {config.population}")
    print(f"  - Grid: {config.elites_grid}")
    print()
    
    # 2. Create adapter
    adapter = RLSyntheticAdapter(
        state_dim=10,
        action_dim=5,
        episode_length=50,
    )
    
    print("✓ RL Synthetic Adapter created")
    print(f"  - State dim: {adapter.state_dim}")
    print(f"  - Action dim: {adapter.action_dim}")
    print()
    
    # 3. Create and run engine
    print("Starting Fibonacci Engine...")
    print()
    
    engine = FibonacciEngine(
        config=config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    # Run evolution
    result = engine.run()
    
    # 4. Display results
    print()
    print("="*60)
    print("Results")
    print("="*60)
    print(f"Best Fitness: {result['archive']['best_fitness']:.6f}")
    print(f"Mean Fitness: {result['archive']['mean_fitness']:.6f}")
    print(f"Elites: {result['archive']['n_elites']}")
    print(f"Coverage: {result['archive']['coverage']:.2%}")
    print()
    
    # Get best solution
    best = engine.archive.get_best()
    print(f"Best Solution:")
    print(f"  - Fitness: {best.fitness:.6f}")
    print(f"  - Descriptor: {best.descriptor}")
    print(f"  - Generation: {best.generation}")
    print()
    
    # Meta-controller recommendation
    recommendation = engine.meta_controller.get_recommendation()
    print(f"Meta-Controller Recommendation:")
    print(f"  - Best Strategy: {recommendation['recommended_arm']}")
    print(f"  - Confidence: {recommendation['confidence']:.2%}")
    print()
    
    # 5. Save outputs
    engine.snapshot("fibonacci_engine/persistence/example_snapshot.json")
    engine.ledger.save("fibonacci_engine/persistence/example_ledger.json")
    
    print("✓ Snapshot saved to: fibonacci_engine/persistence/example_snapshot.json")
    print("✓ Ledger saved to: fibonacci_engine/persistence/example_ledger.json")
    print()
    
    # Verify ledger
    is_valid = engine.ledger.verify()
    print(f"Ledger integrity: {'✓ Valid' if is_valid else '✗ Invalid'}")
    print()
    
    print("="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    main()
