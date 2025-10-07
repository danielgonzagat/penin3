"""
Command-Line Interface for Fibonacci Engine

Provides commands to start, run, monitor, and manage the engine.
"""

import click
import json
import yaml
from pathlib import Path
from typing import Optional
import numpy as np

from fibonacci_engine.core.motor_fibonacci import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters.rl_synthetic import RLSyntheticAdapter
from fibonacci_engine.adapters.supervised_synthetic import SupervisedSyntheticAdapter
from fibonacci_engine.adapters.tool_pipeline import ToolPipelineAdapter


# Global engine instance (for interactive sessions)
_engine: Optional[FibonacciEngine] = None


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Fibonacci Engine - Universal AI Optimization Engine
    
    A state-of-the-art optimization engine inspired by Fibonacci sequence
    and golden ratio, featuring quality-diversity evolution, multi-scale
    spiral search, and adaptive meta-control.
    """
    pass


@cli.command()
@click.option('--cfg', type=click.Path(exists=True), help='Config file (YAML/JSON)')
@click.option('--adapter', type=click.Choice(['rl', 'supervised', 'tool']), 
              default='rl', help='Adapter type')
@click.option('--generations', type=int, default=None, help='Override max generations')
@click.option('--verbose', type=int, default=1, help='Verbosity level (0-2)')
def run(cfg: Optional[str], adapter: str, generations: Optional[int], verbose: int):
    """
    Run the Fibonacci Engine with specified configuration.
    
    Example:
        fib run --adapter rl --generations 60
        fib run --cfg config.yaml
    """
    global _engine
    
    # Load config
    if cfg:
        config_path = Path(cfg)
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        config = FibonacciConfig.from_dict(config_dict)
    else:
        config = FibonacciConfig()
    
    # Override parameters
    if generations:
        config.max_generations = generations
    if verbose is not None:
        config.verbose = verbose
    
    # Select adapter
    if adapter == 'rl':
        adapter_obj = RLSyntheticAdapter()
    elif adapter == 'supervised':
        adapter_obj = SupervisedSyntheticAdapter()
    else:  # tool
        adapter_obj = ToolPipelineAdapter()
    
    # Create engine
    click.echo(f"🌀 Initializing Fibonacci Engine with {adapter} adapter...")
    click.echo(f"   Generations: {config.max_generations}")
    click.echo(f"   Population: {config.population}")
    click.echo(f"   Grid: {config.elites_grid}")
    click.echo()
    
    _engine = FibonacciEngine(
        config=config,
        evaluate_fn=adapter_obj.evaluate,
        descriptor_fn=adapter_obj.descriptor,
        mutate_fn=adapter_obj.mutate,
        cross_fn=adapter_obj.crossover,
        task_sampler=adapter_obj.task_sampler,
    )
    
    # Run
    click.echo("🚀 Starting evolution...")
    result = _engine.run(generations=config.max_generations)
    
    # Display results
    click.echo()
    click.echo("="*60)
    click.echo("✅ Run Complete!")
    click.echo("="*60)
    click.echo(f"Final Generation: {result['generation']}")
    click.echo(f"Best Fitness: {result['archive']['best_fitness']:.6f}")
    click.echo(f"Mean Fitness: {result['archive']['mean_fitness']:.6f}")
    click.echo(f"Elites: {result['archive']['n_elites']}")
    click.echo(f"Coverage: {result['archive']['coverage']:.2%}")
    click.echo(f"Ledger Entries: {result['ledger']['total_entries']}")
    click.echo(f"Mean Gen Time: {result['mean_gen_time']:.3f}s")
    click.echo()
    
    # Save snapshot
    snapshot_path = "fibonacci_engine/persistence/final_snapshot.json"
    _engine.snapshot(snapshot_path)
    click.echo(f"💾 Snapshot saved to: {snapshot_path}")
    
    # Save ledger
    ledger_path = "fibonacci_engine/persistence/ledger.json"
    _engine.ledger.save(ledger_path)
    click.echo(f"📜 Ledger saved to: {ledger_path}")


@cli.command()
@click.option('--cfg', type=click.Path(exists=True), help='Config file (YAML/JSON)')
@click.option('--adapter', type=click.Choice(['rl', 'supervised', 'tool']), 
              default='rl', help='Adapter type')
def start(cfg: Optional[str], adapter: str):
    """
    Start/initialize the engine (for interactive use).
    
    Example:
        fib start --adapter rl
    """
    global _engine
    
    # Load config
    if cfg:
        config_path = Path(cfg)
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        config = FibonacciConfig.from_dict(config_dict)
    else:
        config = FibonacciConfig()
    
    # Select adapter
    if adapter == 'rl':
        adapter_obj = RLSyntheticAdapter()
    elif adapter == 'supervised':
        adapter_obj = SupervisedSyntheticAdapter()
    else:  # tool
        adapter_obj = ToolPipelineAdapter()
    
    _engine = FibonacciEngine(
        config=config,
        evaluate_fn=adapter_obj.evaluate,
        descriptor_fn=adapter_obj.descriptor,
        mutate_fn=adapter_obj.mutate,
        cross_fn=adapter_obj.crossover,
        task_sampler=adapter_obj.task_sampler,
    )
    
    click.echo("✅ Engine initialized successfully!")


@cli.command()
@click.option('--n', type=int, default=1, help='Number of generations to step')
def step(n: int):
    """
    Execute N generation steps.
    
    Example:
        fib step --n 10
    """
    global _engine
    
    if _engine is None:
        click.echo("❌ Engine not initialized. Run 'fib start' first.")
        return
    
    for i in range(n):
        _engine.step()
    
    click.echo(f"✅ Completed {n} generation(s)")


@cli.command()
def status():
    """
    Display current engine status.
    
    Example:
        fib status
    """
    global _engine
    
    if _engine is None:
        click.echo("❌ Engine not initialized.")
        return
    
    status = _engine.get_status()
    
    click.echo("="*60)
    click.echo("📊 Fibonacci Engine Status")
    click.echo("="*60)
    click.echo(f"Generation: {status['generation']}")
    click.echo(f"Running: {status['is_running']}")
    click.echo()
    click.echo("Archive:")
    click.echo(f"  Best Fitness: {status['archive']['best_fitness']:.6f}")
    click.echo(f"  Mean Fitness: {status['archive']['mean_fitness']:.6f}")
    click.echo(f"  Elites: {status['archive']['n_elites']}")
    click.echo(f"  Coverage: {status['archive']['coverage']:.2%}")
    click.echo()
    click.echo("Meta-Controller:")
    click.echo(f"  Total Pulls: {status['meta_controller']['total_pulls']}")
    click.echo(f"  Best Arm: {status['meta_controller']['best_arm']}")
    click.echo()
    click.echo("Rollback Guard:")
    click.echo(f"  Rollbacks Triggered: {status['rollback_guard']['n_rollbacks_triggered']}")
    click.echo()
    click.echo("Ledger:")
    click.echo(f"  Total Entries: {status['ledger']['total_entries']}")
    click.echo(f"  Valid: {status['ledger']['is_valid']}")


@cli.command()
@click.option('--out', type=click.Path(), default='fibonacci_engine/persistence/snapshot.json',
              help='Output path')
def snapshot(out: str):
    """
    Save current engine snapshot.
    
    Example:
        fib snapshot --out snapshots/gen50.json
    """
    global _engine
    
    if _engine is None:
        click.echo("❌ Engine not initialized.")
        return
    
    _engine.snapshot(out)
    click.echo(f"✅ Snapshot saved to: {out}")


@cli.command()
@click.option('--out', type=click.Path(), default='fibonacci_engine/reports/summary.md',
              help='Output path')
def report(out: str):
    """
    Generate summary report.
    
    Example:
        fib report --out reports/summary.md
    """
    global _engine
    
    if _engine is None:
        click.echo("❌ Engine not initialized.")
        return
    
    status = _engine.get_status()
    
    # Create output directory
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report_text = f"""# Fibonacci Engine - Execution Report

## Summary

- **Generation**: {status['generation']}
- **Best Fitness**: {status['archive']['best_fitness']:.6f}
- **Mean Fitness**: {status['archive']['mean_fitness']:.6f}
- **Median Fitness**: {status['archive'].get('median_fitness', 0.0):.6f}
- **Elite Count**: {status['archive']['n_elites']}
- **Coverage**: {status['archive']['coverage']:.2%}

## Archive Statistics

- Total Insertions: {status['archive']['n_insertions']}
- Total Replacements: {status['archive']['n_replacements']}

## Meta-Controller

- Total Pulls: {status['meta_controller']['total_pulls']}
- Best Strategy: {status['meta_controller']['best_arm']}
- Most Used Strategy: {status['meta_controller']['most_pulled_arm']}

### Strategy Performance

"""
    
    for arm_name, arm_data in status['meta_controller']['arms'].items():
        report_text += f"- **{arm_name}**: "
        report_text += f"Mean Reward = {arm_data['mean_reward']:.4f}, "
        report_text += f"Pulls = {arm_data['n_pulls']}\n"
    
    report_text += f"""
## Curriculum

- Total Tasks: {status['curriculum']['total_tasks_sampled']}
- Current Difficulty: {status['curriculum']['current_difficulty']:.2f}

## Rollback Guard

- Rollbacks Triggered: {status['rollback_guard']['n_rollbacks_triggered']}
- Baseline Best: {status['rollback_guard'].get('baseline_best', 'N/A')}

## Ledger

- Total Entries: {status['ledger']['total_entries']}
- Valid: {status['ledger']['is_valid']}

## Performance

- Mean Generation Time: {status['mean_gen_time']:.3f}s

---

Generated by Fibonacci Engine v1.0.0
"""
    
    with open(out_path, 'w') as f:
        f.write(report_text)
    
    click.echo(f"✅ Report saved to: {out}")


@cli.command()
def examples():
    """
    Display example usage commands.
    """
    examples_text = """
🌀 Fibonacci Engine - Example Commands

Basic Usage:
  $ fib run --adapter rl --generations 60
  $ fib run --adapter supervised --generations 100
  $ fib run --adapter tool --generations 50

With Config File:
  $ fib run --cfg myconfig.yaml

Interactive Mode:
  $ fib start --adapter rl
  $ fib step --n 10
  $ fib status
  $ fib snapshot --out gen10.json

Generate Reports:
  $ fib report --out summary.md

Advanced:
  $ fib run --adapter rl --generations 200 --verbose 2
  $ fib snapshot --out snapshots/final.json
  $ fib report --out reports/detailed.md

For more information, see:
  - README.md
  - QUICK_START.md
  - INTEGRATION_GUIDE.md
"""
    click.echo(examples_text)


if __name__ == '__main__':
    cli()
