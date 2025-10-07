# 🚀 Quick Start Guide - Fibonacci Engine

Get the Fibonacci Engine running in less than 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker for containerized deployment

## Step 1: Installation

### From Source

```bash
# Clone repository
git clone https://github.com/fibonacci-engine/fibonacci-engine.git
cd fibonacci-engine

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
fib --help
```

You should see the Fibonacci Engine CLI help message.

## Step 2: Run Your First Evolution

### Option A: CLI (Easiest)

```bash
# Run with default RL adapter for 60 generations
fib run --adapter rl --generations 60
```

This will:
1. Initialize the engine with synthetic RL environment
2. Evolve a population of 48 solutions
3. Track diversity across a 12×12 behavioral grid
4. Save results to `fibonacci_engine/persistence/`

### Option B: Python Script

Create a file `my_first_run.py`:

```python
from fibonacci_engine import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters import RLSyntheticAdapter

# Configure
config = FibonacciConfig(
    max_generations=30,
    population=32,
    verbose=1,
)

# Create adapter
adapter = RLSyntheticAdapter()

# Initialize engine
engine = FibonacciEngine(
    config=config,
    evaluate_fn=adapter.evaluate,
    descriptor_fn=adapter.descriptor,
    mutate_fn=adapter.mutate,
    cross_fn=adapter.crossover,
    task_sampler=adapter.task_sampler,
)

# Run!
result = engine.run()

# Display results
print(f"\n🎉 Evolution Complete!")
print(f"Best Fitness: {result['archive']['best_fitness']:.4f}")
print(f"Elites Found: {result['archive']['n_elites']}")
print(f"Coverage: {result['archive']['coverage']:.1%}")
```

Run it:
```bash
python my_first_run.py
```

## Step 3: Explore Results

### View Summary Report

```bash
fib report --out reports/my_report.md
cat reports/my_report.md
```

### Inspect Archive

Check the saved files in `fibonacci_engine/persistence/`:
- `final_snapshot.json` - Complete engine state
- `ledger.json` - Immutable audit log

### Verify Integrity

```python
from fibonacci_engine.core.worm_ledger import WormLedger

ledger = WormLedger.load("fibonacci_engine/persistence/ledger.json")
print(f"Ledger valid: {ledger.verify()}")
print(f"Total events: {len(ledger)}")
```

## Step 4: Try Different Adapters

### Supervised Learning

```bash
fib run --adapter supervised --generations 50
```

### Tool Pipeline

```bash
fib run --adapter tool --generations 40
```

## Step 5: Interactive Mode

For more control, use interactive mode:

```bash
# Start engine
fib start --adapter rl

# Step through generations manually
fib step --n 5

# Check status
fib status

# Save snapshot
fib snapshot --out my_snapshot.json

# Continue evolution
fib step --n 10
```

## Step 6: Use REST API

### Start Server

```bash
python -m fibonacci_engine.api.rest
```

Server runs at `http://localhost:8000`

### Make API Calls

```bash
# Initialize engine
curl -X POST http://localhost:8000/engine/start \
  -H "Content-Type: application/json" \
  -d '{"adapter": "rl"}'

# Run evolution
curl -X POST http://localhost:8000/engine/run \
  -H "Content-Type: application/json" \
  -d '{"generations": 30}'

# Get status
curl http://localhost:8000/engine/status

# Get best elite
curl http://localhost:8000/elites/best

# Verify ledger
curl http://localhost:8000/ledger/verify
```

### Browse Interactive Docs

Visit `http://localhost:8000/docs` for full API documentation with interactive examples.

## Configuration Options

### Via YAML Config

Create `my_config.yaml`:

```yaml
max_generations: 100
fib_depth: 12
population: 64
elites_grid: [16, 16]
rollback_delta: 0.02
seed: 42
enable_curriculum: true
enable_rollback: true
save_snapshots_every: 10
verbose: 1
```

Run with config:
```bash
fib run --cfg my_config.yaml --adapter rl
```

### Via Python

```python
config = FibonacciConfig(
    max_generations=100,
    fib_depth=12,
    population=64,
    elites_grid=(16, 16),
    rollback_delta=0.02,
    seed=42,
    meta_control_arms=["small", "medium", "large", "adaptive"],
    enable_curriculum=True,
    enable_rollback=True,
    save_snapshots_every=10,
    verbose=1,
)
```

## Understanding the Output

During evolution, you'll see:

```
============================================================
Generation 1
============================================================
  Tasks: 1
  Window: 1
  Budget: explore=19, exploit=13
  Scales: (0.125, 0.25, 0.375)
  Best: 0.5234
  Mean: 0.4891
  Elites: 12
  Coverage: 4.86%
  Improvements: 8/32
  Time: 0.42s
```

**Explanation:**
- **Tasks**: Number of evaluation tasks (grows with Fibonacci sequence)
- **Window**: Current Fibonacci window size
- **Budget**: Split between exploration and exploitation
- **Scales**: Three perturbation magnitudes for spiral search
- **Best/Mean**: Fitness statistics
- **Elites**: Number of unique elite solutions
- **Coverage**: Percentage of behavioral niches filled
- **Improvements**: How many candidates improved the archive
- **Time**: Generation duration

## Next Steps

1. **Custom Adapters**: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) to connect your own system
2. **Advanced Config**: Explore `fibonacci_engine/examples/` for more configurations
3. **Analysis**: Use the snapshot and ledger data for detailed analysis
4. **Visualization**: Build custom visualizations from the archive data

## Troubleshooting

### Issue: Command not found

```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall
pip install -e .
```

### Issue: Import errors

```bash
# Check installation
pip list | grep fibonacci

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Permission denied

```bash
# Check directory permissions
chmod +x fibonacci_engine/examples/run_example.py
```

## Example Output

Successful run should end with:

```
============================================================
✅ Run Complete!
============================================================
Final Generation: 60
Best Fitness: 0.8234
Mean Fitness: 0.7123
Elites: 87
Coverage: 60.42%
Ledger Entries: 245
Mean Gen Time: 0.523s

💾 Snapshot saved to: fibonacci_engine/persistence/final_snapshot.json
📜 Ledger saved to: fibonacci_engine/persistence/ledger.json
```

## Congratulations! 🎉

You've successfully run the Fibonacci Engine! Ready to integrate it with your own system? Check out the [Integration Guide](INTEGRATION_GUIDE.md).

---

**Questions?** See [README.md](README.md) or open an issue on GitHub.
