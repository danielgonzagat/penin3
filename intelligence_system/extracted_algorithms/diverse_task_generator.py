"""
Diverse Task Generator for MAML
Generates 100+ diverse tasks for meta-learning
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import Task from maml_engine
try:
    from .maml_engine import Task
except ImportError:
    from intelligence_system.extracted_algorithms.maml_engine import Task


class DiverseTaskGenerator:
    """
    Generates diverse tasks for meta-learning
    
    Types of tasks:
    1. Classification (N-way K-shot)
    2. Regression (various functions)
    3. Synthetic patterns
    """
    
    def __init__(self, n_tasks_pool: int = 100, seed: int = 42):
        self.n_tasks_pool = n_tasks_pool
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.task_templates = []
        self._initialize_task_pool()
        logger.info(f"🎲 DiverseTaskGenerator initialized: {n_tasks_pool} tasks in pool")
    
    def _initialize_task_pool(self):
        """Initialize pool of task templates"""
        # 50 classification tasks
        for i in range(50):
            n_classes = self.rng.randint(2, 10)
            input_dim = self.rng.choice([28*28, 100, 200, 32*32])
            
            self.task_templates.append({
                'type': 'classification',
                'n_classes': n_classes,
                'input_dim': input_dim,
                'task_id': f'clf_{i}'
            })
        
        # 30 regression tasks
        for i in range(30):
            function_type = self.rng.choice(['linear', 'quadratic', 'sinusoidal', 'exponential'])
            input_dim = self.rng.randint(1, 10)
            
            self.task_templates.append({
                'type': 'regression',
                'function_type': function_type,
                'input_dim': input_dim,
                'task_id': f'reg_{i}'
            })
        
        # 20 pattern recognition tasks
        for i in range(20):
            pattern_type = self.rng.choice(['alternating', 'fibonacci', 'prime', 'geometric'])
            
            self.task_templates.append({
                'type': 'pattern',
                'pattern_type': pattern_type,
                'task_id': f'pat_{i}'
            })
    
    def generate_task(self, template: Dict) -> Task:
        """Generate a concrete task from template"""
        task_type = template['type']
        
        if task_type == 'classification':
            return self._generate_classification_task(template)
        elif task_type == 'regression':
            return self._generate_regression_task(template)
        elif task_type == 'pattern':
            return self._generate_pattern_task(template)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _generate_classification_task(self, template: Dict) -> Task:
        """Generate N-way K-shot classification task"""
        n_classes = template['n_classes']
        input_dim = template['input_dim']
        k_shot = 5  # 5 examples per class in support set
        
        # Support set: K examples per class
        support_x = []
        support_y = []
        
        for class_id in range(n_classes):
            # Generate K synthetic examples for this class
            # Each class has a specific "pattern" in the input space
            class_center = np.random.randn(input_dim) * 0.5
            
            for _ in range(k_shot):
                example = class_center + np.random.randn(input_dim) * 0.1
                support_x.append(example)
                support_y.append(class_id)
        
        # Query set: Different examples from same classes
        query_x = []
        query_y = []
        
        for class_id in range(n_classes):
            class_center = support_x[class_id * k_shot]  # Use support center
            
            for _ in range(5):  # 5 query examples per class
                example = class_center + np.random.randn(input_dim) * 0.15
                query_x.append(example)
                query_y.append(class_id)
        
        # Convert to tensors
        support_x = torch.FloatTensor(np.array(support_x))
        support_y = torch.LongTensor(support_y)
        query_x = torch.FloatTensor(np.array(query_x))
        query_y = torch.LongTensor(query_y)
        
        return Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            task_id=template['task_id']
        )
    
    def _generate_regression_task(self, template: Dict) -> Task:
        """Generate regression task"""
        function_type = template['function_type']
        input_dim = template['input_dim']
        
        # Generate function parameters
        if function_type == 'linear':
            weights = np.random.randn(input_dim)
            bias = np.random.randn()
            func = lambda x: np.dot(x, weights) + bias
        
        elif function_type == 'quadratic':
            a = np.random.randn()
            b = np.random.randn()
            c = np.random.randn()
            func = lambda x: a * (x[0]**2) + b * x[0] + c if input_dim == 1 else np.dot(x, x) * a + b
        
        elif function_type == 'sinusoidal':
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            amp = np.random.uniform(0.5, 2.0)
            func = lambda x: amp * np.sin(freq * x[0] + phase) if input_dim == 1 else amp * np.sin(freq * np.sum(x) + phase)
        
        elif function_type == 'exponential':
            rate = np.random.uniform(0.1, 0.5)
            func = lambda x: np.exp(rate * x[0]) if input_dim == 1 else np.exp(rate * np.sum(x))
        
        # Generate support set
        n_support = 20
        support_x = np.random.randn(n_support, input_dim)
        support_y = np.array([func(x) for x in support_x])
        
        # Generate query set
        n_query = 20
        query_x = np.random.randn(n_query, input_dim)
        query_y = np.array([func(x) for x in query_x])
        
        # Convert to tensors
        support_x = torch.FloatTensor(support_x)
        support_y = torch.FloatTensor(support_y).unsqueeze(-1)
        query_x = torch.FloatTensor(query_x)
        query_y = torch.FloatTensor(query_y).unsqueeze(-1)
        
        return Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            task_id=template['task_id']
        )
    
    def _generate_pattern_task(self, template: Dict) -> Task:
        """Generate pattern recognition task"""
        pattern_type = template['pattern_type']
        
        # Generate sequences following pattern
        n_support = 20
        seq_length = 10
        
        support_sequences = []
        support_labels = []
        
        for _ in range(n_support):
            if pattern_type == 'alternating':
                # Alternating 0, 1, 0, 1, ...
                seq = [i % 2 for i in range(seq_length)]
                next_val = seq_length % 2
            
            elif pattern_type == 'fibonacci':
                # Fibonacci sequence normalized
                fib = [0, 1]
                for i in range(seq_length - 2):
                    fib.append(fib[-1] + fib[-2])
                max_val = max(fib) if max(fib) > 0 else 1
                seq = [f / max_val for f in fib[:seq_length]]
                next_val = fib[seq_length] / max_val if len(fib) > seq_length else 0.5
            
            elif pattern_type == 'prime':
                # Prime numbers normalized
                primes = self._get_primes(seq_length + 10)
                max_val = max(primes) if max(primes) > 0 else 1
                seq = [p / max_val for p in primes[:seq_length]]
                next_val = primes[seq_length] / max_val if len(primes) > seq_length else 0.5
            
            elif pattern_type == 'geometric':
                # Geometric sequence
                ratio = self.rng.uniform(1.1, 2.0)
                start = self.rng.uniform(0.1, 1.0)
                seq = [start * (ratio ** i) for i in range(seq_length)]
                max_val = max(seq) if max(seq) > 0 else 1
                seq = [s / max_val for s in seq]
                next_val = (start * (ratio ** seq_length)) / max_val
            
            # Add noise
            seq = [s + np.random.randn() * 0.05 for s in seq]
            
            support_sequences.append(seq)
            support_labels.append(next_val)
        
        # Generate query set
        query_sequences = []
        query_labels = []
        
        for _ in range(20):
            # Same logic as support
            if pattern_type == 'alternating':
                seq = [i % 2 for i in range(seq_length)]
                next_val = seq_length % 2
            elif pattern_type == 'fibonacci':
                fib = [0, 1]
                for i in range(seq_length - 2):
                    fib.append(fib[-1] + fib[-2])
                max_val = max(fib) if max(fib) > 0 else 1
                seq = [f / max_val for f in fib[:seq_length]]
                next_val = fib[seq_length] / max_val if len(fib) > seq_length else 0.5
            elif pattern_type == 'prime':
                primes = self._get_primes(seq_length + 10)
                max_val = max(primes) if max(primes) > 0 else 1
                seq = [p / max_val for p in primes[:seq_length]]
                next_val = primes[seq_length] / max_val if len(primes) > seq_length else 0.5
            elif pattern_type == 'geometric':
                ratio = self.rng.uniform(1.1, 2.0)
                start = self.rng.uniform(0.1, 1.0)
                seq = [start * (ratio ** i) for i in range(seq_length)]
                max_val = max(seq) if max(seq) > 0 else 1
                seq = [s / max_val for s in seq]
                next_val = (start * (ratio ** seq_length)) / max_val
            
            seq = [s + np.random.randn() * 0.05 for s in seq]
            query_sequences.append(seq)
            query_labels.append(next_val)
        
        # Convert to tensors
        support_x = torch.FloatTensor(support_sequences)
        support_y = torch.FloatTensor(support_labels).unsqueeze(-1)
        query_x = torch.FloatTensor(query_sequences)
        query_y = torch.FloatTensor(query_labels).unsqueeze(-1)
        
        return Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            task_id=template['task_id']
        )
    
    def _get_primes(self, n):
        """Get first n prime numbers"""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    def generate_batch(self, batch_size: int = 10) -> List[Task]:
        """Generate a batch of diverse tasks"""
        # Sample from task pool
        templates = self.rng.sample(self.task_templates, min(batch_size, len(self.task_templates)))
        
        # Generate concrete tasks
        tasks = []
        for template in templates:
            try:
                task = self.generate_task(template)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to generate task {template['task_id']}: {e}")
        
        return tasks
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about task pool"""
        type_counts = {}
        for template in self.task_templates:
            task_type = template['type']
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        return {
            'total_tasks': len(self.task_templates),
            'type_distribution': type_counts
        }


# Convenience function for V7 integration
def create_diverse_tasks(n_tasks: int = 10) -> List[Task]:
    """Create diverse tasks for MAML training"""
    generator = DiverseTaskGenerator(n_tasks_pool=100)
    return generator.generate_batch(batch_size=n_tasks)


if __name__ == "__main__":
    # Test task generation
    generator = DiverseTaskGenerator(n_tasks_pool=100)
    
    print("📊 Task Pool Statistics:")
    stats = generator.get_task_statistics()
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Distribution: {stats['type_distribution']}")
    
    print("\n🎲 Generating sample batch...")
    tasks = generator.generate_batch(batch_size=5)
    print(f"   Generated {len(tasks)} tasks")
    
    for task in tasks:
        print(f"   - {task.task_id}: support shape={task.support_x.shape}, query shape={task.query_x.shape}")