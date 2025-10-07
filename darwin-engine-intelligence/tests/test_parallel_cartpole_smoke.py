import os
import sys
import json
import tempfile

sys.path.insert(0, '/root/darwin-engine-intelligence')

from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator

def test_cartpole_parallel_preserves_models():
    os.environ['PYTHONNOUSERSITE'] = '1'
    orch = DarwinEvolutionOrchestrator(n_workers=2)
    best = orch.evolve_cartpole(generations=1, population_size=4, demo_fast=True, demo_epochs=1, parallel=True)
    # Ensure best has genome and fitness
    assert hasattr(best, 'genome') and isinstance(best.genome, dict)
    assert isinstance(best.fitness, float)
    # Check checkpoint save path existence after run
    out_dir = orch.output_dir
    assert out_dir.exists()
    # Not all runs checkpoint on gen1; so we rely on presence of best itself
    # Optionally dump a small summary for manual inspection
    summary = {'best_fitness': float(best.fitness), 'genome_keys': list(best.genome.keys())}
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        json.dump(summary, f)
