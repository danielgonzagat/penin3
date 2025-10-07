# 🌀 Motor Fibonacci - Universal AI Optimization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Motor Fibonacci** is a state-of-the-art, universal AI optimization engine inspired by the Fibonacci sequence and golden ratio. It provides plug-and-play integration with any existing system, significantly enhancing functional intelligence through quality-diversity evolution, multi-scale spiral search, and adaptive meta-control.

## ✨ Key Features

### 🧬 Core Capabilities
- **Fibonacci Scheduling**: Evaluation budget grows harmonically following the Fibonacci sequence
- **Golden Ratio Mixing (Φ)**: Balanced exploration/exploitation using φ ≈ 1.618
- **Multi-Scale Spiral Search**: Perturbations across three harmonically-related scales
- **Quality-Diversity (MAP-Elites)**: Maintains diverse population across behavioral space
- **Adaptive Meta-Control**: UCB bandit algorithm selects optimal search strategies
- **Curriculum Learning**: Progressive task difficulty following Fibonacci windows

### 🛡️ Safety & Auditability
- **WORM Ledger**: Immutable hash-chain event log for complete auditability
- **Automatic Rollback**: Detects and prevents performance regressions
- **Transactional Snapshots**: Safe checkpointing and recovery
- **Sandbox Execution**: No host file system modification by default

### 🔌 Universal Integration
- **Adapter Pattern**: Five simple functions connect to any system
- **Zero Assumptions**: Host-agnostic architecture
- **Non-Invasive**: Operates via provided interfaces only
- **Plug-and-Play**: Drop-in enhancement for existing systems

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fibonacci-engine/fibonacci-engine.git
cd fibonacci-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -e .

# Verify installation
fib --help
```

### Run Your First Evolution

```bash
# Run with RL synthetic adapter
fib run --adapter rl --generations 60

# Run with supervised learning adapter
fib run --adapter supervised --generations 100

# Run with custom config
fib run --cfg examples/config_rl.yaml
```

### Python API

```python
from fibonacci_engine import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters import RLSyntheticAdapter

# Create configuration
config = FibonacciConfig(
    max_generations=60,
    population=48,
    elites_grid=(12, 12),
    seed=42,
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

# Run evolution
result = engine.run()

print(f"Best Fitness: {result['archive']['best_fitness']:.6f}")
print(f"Coverage: {result['archive']['coverage']:.2%}")
```

### REST API

```bash
# Start the server
python -m fibonacci_engine.api.rest

# In another terminal, use the API
curl -X POST http://localhost:8000/engine/start \
  -H "Content-Type: application/json" \
  -d '{"adapter": "rl"}'

curl -X POST http://localhost:8000/engine/run \
  -H "Content-Type: application/json" \
  -d '{"generations": 60}'

curl http://localhost:8000/engine/status
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in 5 minutes
- **[Integration Guide](INTEGRATION_GUIDE.md)** - Connect your own system
- **[API Documentation](http://localhost:8000/docs)** - REST API reference (when server is running)
- **[Examples](fibonacci_engine/examples/)** - Sample configurations and scripts

## 🧩 Architecture

### Core Components

1. **FibonacciEngine**: Main orchestrator coordinating all subsystems
2. **MAPElites**: Quality-diversity archive maintaining diverse elite solutions
3. **MetaController**: UCB bandit for adaptive strategy selection
4. **FibonacciCurriculum**: Progressive task sampling
5. **WormLedger**: Immutable event log with hash-chain verification
6. **RollbackGuard**: Automatic regression detection and protection

### Universal Adapters

Connect any system by implementing five functions:

```python
def evaluate_fn(params, tasks) -> dict:
    """Evaluate parameters on tasks. Return metrics with 'fitness' key."""
    pass

def descriptor_fn(params, metrics) -> list:
    """Compute behavioral descriptor. Return list of values in [0, 1]."""
    pass

def mutate_fn(params, magnitude) -> params:
    """Mutate parameters with given magnitude."""
    pass

def cross_fn(params_a, params_b) -> params:
    """Crossover two parameter sets."""
    pass

def task_sampler(n, difficulty) -> list:
    """Sample n tasks with given difficulty."""
    pass
```

## 🔬 Example Adapters

### 1. Reinforcement Learning
```python
from fibonacci_engine.adapters import RLSyntheticAdapter
adapter = RLSyntheticAdapter(state_dim=10, action_dim=5)
```

### 2. Supervised Learning
```python
from fibonacci_engine.adapters import SupervisedSyntheticAdapter
adapter = SupervisedSyntheticAdapter(input_dim=20, output_dim=1)
```

### 3. Tool Pipeline
```python
from fibonacci_engine.adapters import ToolPipelineAdapter
adapter = ToolPipelineAdapter(n_transformations=8)
```

## 📊 Results & Metrics

The engine tracks comprehensive metrics:

- **Best Fitness**: Global best performance
- **Mean/Median Fitness**: Archive quality statistics
- **Coverage**: Percentage of behavioral niches filled
- **Elite Count**: Number of unique elite solutions
- **Meta-Controller Stats**: Strategy performance
- **Ledger Integrity**: Audit trail verification

Generate reports:
```bash
fib report --out reports/summary.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fibonacci_engine --cov-report=html

# Run specific test suite
pytest fibonacci_engine/tests/test_integration.py
```

## 🐳 Docker Support

```bash
# Build image
docker build -t fibonacci-engine .

# Run container
docker run -it fibonacci-engine fib run --adapter rl --generations 60
```

## 🎯 Use Cases

- **Hyperparameter Optimization**: Evolve optimal configurations
- **Neural Architecture Search**: Discover novel model architectures
- **Reinforcement Learning**: Learn diverse behavioral policies
- **Automated Machine Learning**: Self-improving ML pipelines
- **Multi-Objective Optimization**: Balance competing objectives
- **Creative AI**: Generate diverse, high-quality outputs

## 🌟 Key Innovations

1. **Harmonic Expansion**: Growth follows Fibonacci sequence for balanced scaling
2. **Φ-Mixing**: Golden ratio ensures optimal exploration/exploitation balance
3. **Spiral Search**: Multi-scale perturbations from local to global
4. **Quality-Diversity**: Prioritizes both performance AND diversity
5. **Self-Adaptive**: Meta-controller learns optimal strategies during runtime
6. **Provably Auditable**: WORM ledger provides cryptographic verification

## 📖 Theoretical Foundation

The Fibonacci Engine is based on:

- **Fibonacci Sequence**: F(n) = F(n-1) + F(n-2), starting with F(1) = F(2) = 1
- **Golden Ratio**: φ = (1 + √5) / 2 ≈ 1.618
- **MAP-Elites Algorithm**: Mouret & Clune (2015)
- **UCB Bandit**: Auer et al. (2002)
- **Curriculum Learning**: Bengio et al. (2009)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Inspired by the mathematical beauty of the Fibonacci sequence and the universal principles of harmonic growth found in nature.

## 📞 Support

- **Documentation**: [Quick Start](QUICK_START.md) | [Integration Guide](INTEGRATION_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/fibonacci-engine/fibonacci-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fibonacci-engine/fibonacci-engine/discussions)

---

**Built with ❤️ and φ (golden ratio)**
