#!/bin/bash
# Quick installation and test script

echo "🌀 Motor Fibonacci - Installation Test"
echo "======================================"
echo ""

echo "1️⃣  Installing dependencies..."
pip install -q -e . 2>&1 | grep -i "success\|complete\|installed" || echo "Installation attempted"

echo ""
echo "2️⃣  Testing CLI..."
export PATH="/home/ubuntu/.local/bin:$PATH"
fib --version 2>/dev/null || fib --help | head -5

echo ""
echo "3️⃣  Testing Python import..."
python3 -c "from fibonacci_engine import FibonacciEngine, FibonacciConfig; print('✅ Import successful')"

echo ""
echo "4️⃣  Running quick test..."
python3 -c "
from fibonacci_engine import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters import RLSyntheticAdapter
import numpy as np

np.random.seed(42)
config = FibonacciConfig(max_generations=2, population=8, verbose=0)
adapter = RLSyntheticAdapter()
engine = FibonacciEngine(config, adapter.evaluate, adapter.descriptor, adapter.mutate, adapter.crossover, adapter.task_sampler)
result = engine.run()
print(f'✅ Quick test passed - Best fitness: {result[\"archive\"][\"best_fitness\"]:.4f}')
"

echo ""
echo "======================================"
echo "✅ Installation test complete!"
echo ""
echo "Next steps:"
echo "  fib run --adapter rl --generations 30"
echo "  python3 fibonacci_engine/examples/run_example.py"
echo ""

