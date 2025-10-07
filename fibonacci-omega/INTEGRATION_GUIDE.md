# 🔌 Integration Guide - Fibonacci Engine

This guide shows you how to integrate the Fibonacci Engine with your own system, regardless of domain or architecture.

## Table of Contents

1. [Overview](#overview)
2. [The Adapter Pattern](#the-adapter-pattern)
3. [Step-by-Step Integration](#step-by-step-integration)
4. [Real-World Examples](#real-world-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Overview

The Fibonacci Engine connects to any system via **five adapter functions**:

1. **evaluate_fn**: Evaluate solution quality
2. **descriptor_fn**: Compute behavioral descriptor
3. **mutate_fn**: Generate variations
4. **cross_fn**: Combine solutions
5. **task_sampler**: Generate evaluation tasks

These functions form a **universal interface** that makes the engine completely host-agnostic.

## The Adapter Pattern

### Required Interface

```python
from typing import Any, Dict, List

def evaluate_fn(params: Any, tasks: List[Any]) -> Dict[str, float]:
    """
    Evaluate parameters on given tasks.
    
    Args:
        params: Solution parameters (any type your system uses)
        tasks: List of evaluation tasks
        
    Returns:
        Dictionary with at least 'fitness' key (float)
    """
    pass

def descriptor_fn(params: Any, metrics: Dict[str, float]) -> List[float]:
    """
    Compute behavioral descriptor for a solution.
    
    Args:
        params: Solution parameters
        metrics: Evaluation metrics from evaluate_fn
        
    Returns:
        List of values in [0, 1] representing behavior dimensions
    """
    pass

def mutate_fn(params: Any, magnitude: float) -> Any:
    """
    Create a mutated variant of parameters.
    
    Args:
        params: Original parameters
        magnitude: Mutation strength (typically 0.0 to 1.0)
        
    Returns:
        Mutated parameters (same type as input)
    """
    pass

def cross_fn(params_a: Any, params_b: Any) -> Any:
    """
    Combine two parameter sets via crossover.
    
    Args:
        params_a: First parent
        params_b: Second parent
        
    Returns:
        Child parameters (same type as parents)
    """
    pass

def task_sampler(n: int, difficulty: float) -> List[Any]:
    """
    Sample evaluation tasks.
    
    Args:
        n: Number of tasks to sample
        difficulty: Difficulty level in [0, 1]
        
    Returns:
        List of n tasks
    """
    pass
```

## Step-by-Step Integration

### Example: Neural Network Hyperparameter Optimization

Let's integrate the engine to optimize neural network hyperparameters.

#### Step 1: Define Your Parameter Space

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class NNParams:
    learning_rate: float
    num_layers: int
    hidden_size: int
    dropout: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for manipulation."""
        return np.array([
            self.learning_rate,
            float(self.num_layers),
            float(self.hidden_size),
            self.dropout,
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'NNParams':
        """Create from numpy array."""
        return cls(
            learning_rate=max(1e-5, min(1.0, arr[0])),
            num_layers=max(1, min(10, int(arr[1]))),
            hidden_size=max(16, min(1024, int(arr[2]))),
            dropout=max(0.0, min(0.9, arr[3])),
        )
```

#### Step 2: Implement evaluate_fn

```python
def evaluate_nn_params(params: NNParams, tasks: List[Dict]) -> Dict[str, float]:
    """
    Evaluate neural network with given hyperparameters.
    """
    total_accuracy = 0.0
    total_loss = 0.0
    
    for task in tasks:
        # Get dataset for this task
        X_train, y_train = task['train_data']
        X_val, y_val = task['val_data']
        
        # Build and train model with params
        model = build_model(params)
        history = train_model(
            model, 
            X_train, y_train,
            X_val, y_val,
            learning_rate=params.learning_rate,
            epochs=10,  # Quick training
        )
        
        # Evaluate
        val_accuracy = history['val_accuracy'][-1]
        val_loss = history['val_loss'][-1]
        
        total_accuracy += val_accuracy
        total_loss += val_loss
    
    mean_accuracy = total_accuracy / len(tasks)
    mean_loss = total_loss / len(tasks)
    
    return {
        'fitness': mean_accuracy,  # Required key
        'accuracy': mean_accuracy,
        'loss': mean_loss,
        'convergence_speed': len(history['val_loss']),
    }

def build_model(params: NNParams):
    """Build a simple neural network with given params."""
    import tensorflow as tf
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(784,)))  # MNIST-like
    
    for _ in range(params.num_layers):
        model.add(tf.keras.layers.Dense(params.hidden_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(params.dropout))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def train_model(model, X_train, y_train, X_val, y_val, learning_rate, epochs):
    """Quick training."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )
    
    return history.history
```

#### Step 3: Implement descriptor_fn

```python
def compute_nn_descriptor(params: NNParams, metrics: Dict[str, float]) -> List[float]:
    """
    Compute 2D behavioral descriptor.
    
    Dimensions:
    1. Model complexity (normalized)
    2. Generalization (accuracy/loss ratio)
    """
    # Complexity: based on number of parameters
    complexity = (params.num_layers * params.hidden_size) / 10000
    complexity = min(1.0, complexity)
    
    # Generalization: how well it generalizes
    accuracy = metrics.get('accuracy', 0.5)
    loss = metrics.get('loss', 1.0)
    generalization = accuracy / (1.0 + loss)
    generalization = min(1.0, generalization)
    
    return [complexity, generalization]
```

#### Step 4: Implement mutate_fn

```python
def mutate_nn_params(params: NNParams, magnitude: float) -> NNParams:
    """
    Mutate hyperparameters with Gaussian noise.
    """
    arr = params.to_array()
    
    # Add Gaussian noise scaled by magnitude
    noise = np.random.randn(len(arr)) * magnitude
    mutated_arr = arr + noise
    
    # Convert back and clip to valid ranges
    return NNParams.from_array(mutated_arr)
```

#### Step 5: Implement cross_fn

```python
def crossover_nn_params(params_a: NNParams, params_b: NNParams) -> NNParams:
    """
    Crossover two parameter sets using golden ratio blending.
    """
    arr_a = params_a.to_array()
    arr_b = params_b.to_array()
    
    # Blend with golden ratio
    alpha = 0.618  # φ - 1
    child_arr = alpha * arr_a + (1 - alpha) * arr_b
    
    return NNParams.from_array(child_arr)
```

#### Step 6: Implement task_sampler

```python
def sample_nn_tasks(n: int, difficulty: float) -> List[Dict]:
    """
    Sample training tasks with varying difficulty.
    """
    from tensorflow.keras.datasets import mnist
    
    # Load dataset once
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    tasks = []
    for i in range(n):
        # Difficulty controls dataset size and noise
        base_size = int(1000 + difficulty * 4000)  # 1k to 5k samples
        
        # Random subset
        indices = np.random.choice(len(X_train_full), base_size, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
        
        # Add noise based on difficulty
        noise_level = difficulty * 0.1
        X_train = X_train + np.random.randn(*X_train.shape) * noise_level
        
        # Validation set
        val_size = int(base_size * 0.2)
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train = X_train[val_size:]
        y_train = y_train[val_size:]
        
        tasks.append({
            'id': i,
            'train_data': (X_train, y_train),
            'val_data': (X_val, y_val),
            'difficulty': difficulty,
        })
    
    return tasks
```

#### Step 7: Run the Engine

```python
from fibonacci_engine import FibonacciEngine, FibonacciConfig

# Configuration
config = FibonacciConfig(
    max_generations=50,
    population=32,
    elites_grid=(10, 10),
    seed=42,
    verbose=1,
)

# Initial parameters
initial_params = NNParams(
    learning_rate=0.001,
    num_layers=3,
    hidden_size=128,
    dropout=0.2,
)

# Create engine
engine = FibonacciEngine(
    config=config,
    evaluate_fn=evaluate_nn_params,
    descriptor_fn=compute_nn_descriptor,
    mutate_fn=mutate_nn_params,
    cross_fn=crossover_nn_params,
    task_sampler=sample_nn_tasks,
    initial_params=initial_params,
)

# Run optimization
result = engine.run()

# Get best hyperparameters
best = engine.archive.get_best()
print(f"Best accuracy: {best.fitness:.4f}")
print(f"Best params: {best.params}")
```

## Real-World Examples

### Example 1: Prompt Engineering for LLMs

```python
class PromptParams:
    def __init__(self, template: str, examples: List[str], temperature: float):
        self.template = template
        self.examples = examples
        self.temperature = temperature

def evaluate_prompt(params: PromptParams, tasks: List[Dict]) -> Dict[str, float]:
    """Evaluate prompt performance on language tasks."""
    scores = []
    for task in tasks:
        prompt = params.template.format(
            examples="\n".join(params.examples),
            query=task['query']
        )
        
        response = llm_api_call(prompt, temperature=params.temperature)
        score = evaluate_response(response, task['expected'])
        scores.append(score)
    
    return {
        'fitness': np.mean(scores),
        'variance': np.std(scores),
        'cost': len(prompt.split()),
    }

def mutate_prompt(params: PromptParams, magnitude: float) -> PromptParams:
    """Mutate prompt by modifying template or examples."""
    if np.random.rand() < magnitude:
        # Modify template
        new_template = slight_rephrase(params.template)
    else:
        new_template = params.template
    
    # Mutate examples
    new_examples = params.examples.copy()
    if np.random.rand() < magnitude:
        new_examples[np.random.randint(len(new_examples))] = generate_new_example()
    
    # Mutate temperature
    new_temp = params.temperature + np.random.randn() * magnitude * 0.3
    new_temp = max(0.0, min(2.0, new_temp))
    
    return PromptParams(new_template, new_examples, new_temp)

# descriptor_fn, cross_fn, task_sampler follow similar patterns
```

### Example 2: Robot Control Policy

```python
def evaluate_policy(params: np.ndarray, tasks: List[Dict]) -> Dict[str, float]:
    """Evaluate robot control policy."""
    total_reward = 0.0
    for task in tasks:
        env = create_robot_env(task['config'])
        obs = env.reset()
        
        episode_reward = 0.0
        for step in range(task['max_steps']):
            action = policy_network(params, obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        
        total_reward += episode_reward
    
    return {
        'fitness': total_reward / len(tasks),
        'avg_steps': np.mean([task['steps'] for task in tasks]),
    }

def descriptor_robot(params: np.ndarray, metrics: Dict) -> List[float]:
    """Behavioral descriptor for robot: exploration vs exploitation."""
    # Dimension 1: How much area covered (exploration)
    exploration = compute_coverage_metric(params)
    
    # Dimension 2: Efficiency (exploitation)
    efficiency = metrics['fitness'] / metrics['avg_steps']
    
    return [exploration, min(1.0, efficiency)]
```

## Best Practices

### 1. Normalize Fitness

Always normalize fitness to a reasonable range:

```python
def evaluate_fn(params, tasks):
    raw_score = compute_raw_score(params, tasks)
    
    # Normalize to [0, 1]
    normalized = (raw_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    normalized = max(0.0, min(1.0, normalized))
    
    return {'fitness': normalized, 'raw_score': raw_score}
```

### 2. Choose Good Descriptors

Good descriptors capture **behavioral** differences, not just performance:

```python
# ✅ Good: Behavioral dimensions
def descriptor_fn(params, metrics):
    return [
        behavioral_dimension_1(params),  # e.g., strategy type
        behavioral_dimension_2(params),  # e.g., risk profile
    ]

# ❌ Bad: Just fitness decomposition
def descriptor_fn(params, metrics):
    return [
        metrics['fitness'],
        metrics['fitness'] * 2,
    ]
```

### 3. Handle Failures Gracefully

```python
def evaluate_fn(params, tasks):
    try:
        score = run_evaluation(params, tasks)
        return {'fitness': score}
    except Exception as e:
        # Return low fitness instead of crashing
        logging.warning(f"Evaluation failed: {e}")
        return {'fitness': 0.0, 'error': str(e)}
```

### 4. Use Curriculum Wisely

```python
def task_sampler(n, difficulty):
    """Progressive curriculum."""
    if difficulty < 0.3:
        # Easy: simple cases
        return generate_simple_tasks(n)
    elif difficulty < 0.7:
        # Medium: realistic cases
        return generate_realistic_tasks(n)
    else:
        # Hard: edge cases and adversarial
        return generate_challenging_tasks(n)
```

### 5. Save Intermediate Results

```python
# Configure regular snapshots
config = FibonacciConfig(
    save_snapshots_every=10,  # Save every 10 generations
    # ...
)

# Or manually
if generation % 10 == 0:
    engine.snapshot(f"snapshots/gen_{generation}.json")
```

## Troubleshooting

### Issue: Poor Quality-Diversity

**Symptom**: Archive has few elites, low coverage

**Solution**: Improve descriptor function to capture more behavioral variation

```python
# Add more meaningful behavioral dimensions
def descriptor_fn(params, metrics):
    return [
        compute_strategy_type(params),      # Discrete behavioral mode
        compute_risk_profile(params),       # Continuous spectrum
        compute_complexity_measure(params), # Structural property
    ]
```

### Issue: Slow Evaluation

**Symptom**: Each generation takes too long

**Solution**: Parallelize evaluations

```python
from concurrent.futures import ProcessPoolExecutor

def evaluate_fn_parallel(params, tasks):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_single, params, task) for task in tasks]
        results = [f.result() for f in futures]
    
    return aggregate_results(results)
```

### Issue: Unstable Evolution

**Symptom**: Fitness oscillates wildly

**Solution**: Enable rollback guard and adjust mutation scale

```python
config = FibonacciConfig(
    enable_rollback=True,
    rollback_delta=0.05,  # Trigger rollback on 5% drop
    # ...
)

def mutate_fn(params, magnitude):
    # Use smaller mutations
    scale = magnitude * 0.1  # Scale down
    return params + np.random.randn(*params.shape) * scale
```

## Complete Minimal Example

Here's a complete minimal adapter:

```python
import numpy as np
from fibonacci_engine import FibonacciEngine, FibonacciConfig

# 1. Define parameter type
ParamsType = np.ndarray  # Simple: just a numpy array

# 2. Evaluation
def evaluate(params: ParamsType, tasks) -> dict:
    scores = [np.dot(params, task['vector']) for task in tasks]
    return {'fitness': np.mean(scores)}

# 3. Descriptor
def descriptor(params: ParamsType, metrics: dict) -> list:
    return [
        np.std(params),  # Diversity
        np.mean(params),  # Bias
    ]

# 4. Mutation
def mutate(params: ParamsType, magnitude: float) -> ParamsType:
    return params + np.random.randn(len(params)) * magnitude

# 5. Crossover
def crossover(params_a: ParamsType, params_b: ParamsType) -> ParamsType:
    return (params_a + params_b) / 2

# 6. Task sampler
def sample_tasks(n: int, difficulty: float) -> list:
    return [{'vector': np.random.randn(10)} for _ in range(n)]

# Run
engine = FibonacciEngine(
    config=FibonacciConfig(max_generations=20),
    evaluate_fn=evaluate,
    descriptor_fn=descriptor,
    mutate_fn=mutate,
    cross_fn=crossover,
    task_sampler=sample_tasks,
    initial_params=np.random.randn(10),
)

result = engine.run()
print(f"Best: {result['archive']['best_fitness']:.4f}")
```

## Next Steps

1. Start with a minimal adapter (like above)
2. Gradually add complexity as you understand the behavior
3. Experiment with different descriptor functions
4. Monitor the ledger to understand what's happening
5. Use snapshots for analysis and debugging

## Questions?

- Check the [README](README.md) for general information
- See [QUICK_START.md](QUICK_START.md) for basic usage
- Review example adapters in `fibonacci_engine/adapters/`
- Open an issue on GitHub for specific problems

---

**Happy Integrating! 🔌**
