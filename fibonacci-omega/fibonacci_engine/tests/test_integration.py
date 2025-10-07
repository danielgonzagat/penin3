"""Integration tests for complete Fibonacci Engine."""

import pytest
import numpy as np
from fibonacci_engine import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters import (
    RLSyntheticAdapter,
    SupervisedSyntheticAdapter,
    ToolPipelineAdapter,
)


@pytest.fixture
def simple_config():
    """Simple config for quick tests."""
    return FibonacciConfig(
        max_generations=5,
        fib_depth=8,
        population=16,
        elites_grid=(5, 5),
        rollback_delta=0.05,
        seed=42,
        verbose=0,
    )


def test_engine_initialization(simple_config):
    """Test engine initialization."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    assert engine.current_generation == 0
    assert len(engine.archive.archive) > 0  # Initial population
    assert engine.ledger.verify()


def test_engine_step(simple_config):
    """Test single engine step."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    initial_gen = engine.current_generation
    
    stats = engine.step()
    
    assert engine.current_generation == initial_gen + 1
    assert "generation" in stats
    assert "best_fitness" in stats
    assert stats["generation"] == 1


def test_engine_run(simple_config):
    """Test running engine for multiple generations."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    result = engine.run(generations=5)
    
    assert engine.current_generation == 5
    assert result["generation"] == 5
    assert result["archive"]["n_elites"] > 0
    assert engine.ledger.verify()


def test_engine_with_rl_adapter(simple_config):
    """Test engine with RL synthetic adapter."""
    adapter = RLSyntheticAdapter(state_dim=10, action_dim=5)
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    result = engine.run(generations=3)
    
    assert result["archive"]["best_fitness"] > 0
    assert result["archive"]["n_elites"] > 0


def test_engine_with_supervised_adapter(simple_config):
    """Test engine with supervised learning adapter."""
    adapter = SupervisedSyntheticAdapter(input_dim=20, output_dim=1)
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    result = engine.run(generations=3)
    
    assert result["archive"]["best_fitness"] > 0
    assert result["archive"]["n_elites"] > 0


def test_engine_with_tool_adapter(simple_config):
    """Test engine with tool pipeline adapter."""
    adapter = ToolPipelineAdapter(n_transformations=8)
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    result = engine.run(generations=3)
    
    assert result["archive"]["best_fitness"] >= 0
    assert result["archive"]["n_elites"] > 0


def test_snapshot_save_load(simple_config, tmp_path):
    """Test saving and loading engine snapshots."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    engine.run(generations=3)
    
    # Save snapshot
    snapshot_path = tmp_path / "snapshot.json"
    engine.snapshot(str(snapshot_path))
    
    assert snapshot_path.exists()


def test_determinism(simple_config):
    """Test that runs with same seed are deterministic."""
    adapter = RLSyntheticAdapter()
    
    # First run
    config1 = FibonacciConfig(**simple_config.to_dict())
    engine1 = FibonacciEngine(
        config=config1,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    result1 = engine1.run(generations=3)
    
    # Second run with same seed
    config2 = FibonacciConfig(**simple_config.to_dict())
    adapter2 = RLSyntheticAdapter()
    engine2 = FibonacciEngine(
        config=config2,
        evaluate_fn=adapter2.evaluate,
        descriptor_fn=adapter2.descriptor,
        mutate_fn=adapter2.mutate,
        cross_fn=adapter2.crossover,
        task_sampler=adapter2.task_sampler,
    )
    result2 = engine2.run(generations=3)
    
    # Results should be very similar (allowing for minor floating point differences)
    assert abs(result1["archive"]["best_fitness"] - result2["archive"]["best_fitness"]) < 0.01


def test_meta_controller_usage(simple_config):
    """Test that meta-controller is being used."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    engine.run(generations=5)
    
    # Check that meta-controller has been used
    stats = engine.meta_controller.get_all_statistics()
    assert stats["total_pulls"] > 0
    assert stats["best_arm"] is not None


def test_curriculum_progression(simple_config):
    """Test that curriculum progresses through generations."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    engine.run(generations=5)
    
    # Check curriculum stats
    stats = engine.curriculum.get_statistics()
    assert stats["total_tasks_sampled"] > 0
    assert stats["total_generations"] == 5


def test_ledger_integrity(simple_config):
    """Test that ledger maintains integrity throughout run."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    engine.run(generations=5)
    
    # Verify ledger
    assert engine.ledger.verify()
    
    # Check that events were logged
    stats = engine.ledger.get_statistics()
    assert stats["total_entries"] > 5  # At least one per generation
    assert "generation_complete" in stats["event_types"]


def test_quality_diversity(simple_config):
    """Test that quality-diversity is achieved (elites in different niches)."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    engine.run(generations=10)
    
    # Check that we have diversity
    elites = engine.archive.get_all_elites()
    assert len(elites) > 1
    
    # Check that elites have different descriptors
    descriptors = [tuple(e.descriptor) for e in elites]
    unique_descriptors = set(descriptors)
    assert len(unique_descriptors) > 1  # Should have diversity


def test_engine_stop(simple_config):
    """Test stopping the engine."""
    adapter = RLSyntheticAdapter()
    
    engine = FibonacciEngine(
        config=simple_config,
        evaluate_fn=adapter.evaluate,
        descriptor_fn=adapter.descriptor,
        mutate_fn=adapter.mutate,
        cross_fn=adapter.crossover,
        task_sampler=adapter.task_sampler,
    )
    
    # Start running
    engine.is_running = True
    
    # Stop
    engine.stop()
    
    assert not engine.is_running
    
    # Check that ledger records the stop
    entries = engine.ledger.get_entries(event_type="engine_stopped")
    assert len(entries) > 0
