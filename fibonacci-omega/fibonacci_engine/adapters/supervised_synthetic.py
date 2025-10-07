"""
Supervised Synthetic Adapter: Classification/Regression Toy Problem

A simple synthetic supervised learning problem for testing the Fibonacci Engine.
"""

import numpy as np
from typing import Dict, List, Any


class SupervisedSyntheticAdapter:
    """
    Adapter for synthetic supervised learning problems.
    
    Task: Learn a linear classifier/regressor.
    - Input: n-dimensional feature vector
    - Output: scalar or class label
    - Loss: MSE for regression, accuracy for classification
    
    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output (1 for regression, K for classification).
        task_type: "regression" or "classification".
        n_samples_per_task: Number of samples per task.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        output_dim: int = 1,
        task_type: str = "regression",
        n_samples_per_task: int = 100,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.n_samples_per_task = n_samples_per_task
    
    def evaluate(self, params: np.ndarray, tasks: List[Any]) -> Dict[str, float]:
        """
        Evaluate model parameters on supervised tasks.
        
        Args:
            params: Model parameters (weights and bias).
            tasks: List of task specifications.
            
        Returns:
            Dictionary with 'fitness' and other metrics.
        """
        # Reshape params to weight matrix + bias
        expected_size = self.input_dim * self.output_dim + self.output_dim
        if isinstance(params, np.ndarray):
            if params.size != expected_size:
                params = np.random.randn(expected_size) * 0.1
        else:
            params = np.random.randn(expected_size) * 0.1
        
        weights = params[:-self.output_dim].reshape(self.output_dim, self.input_dim)
        bias = params[-self.output_dim:]
        
        total_loss = 0.0
        task_losses = []
        
        for task in tasks:
            # Generate data for this task
            if isinstance(task, dict):
                noise = task.get("noise", 0.1)
                target_weights = np.array(task["target"][:expected_size])
            else:
                noise = 0.1
                target_weights = np.random.randn(expected_size) * 0.5
            
            # Generate samples
            X = np.random.randn(self.n_samples_per_task, self.input_dim)
            
            # True target
            true_w = target_weights[:-self.output_dim].reshape(
                self.output_dim, self.input_dim
            )
            true_b = target_weights[-self.output_dim:]
            y_true = (X @ true_w.T + true_b) + np.random.randn(
                self.n_samples_per_task, self.output_dim
            ) * noise
            
            # Predicted
            y_pred = X @ weights.T + bias
            
            # Loss
            if self.task_type == "regression":
                loss = np.mean((y_true - y_pred) ** 2)
            else:  # classification
                # Simplified: use MSE as proxy
                loss = np.mean((y_true - y_pred) ** 2)
            
            task_losses.append(loss)
            total_loss += loss
        
        mean_loss = total_loss / len(tasks) if tasks else 1.0
        
        # Fitness: inverse of loss
        fitness = 1.0 / (1.0 + mean_loss)
        
        return {
            "fitness": fitness,
            "mean_loss": mean_loss,
            "std_loss": np.std(task_losses) if task_losses else 0.0,
            "min_loss": np.min(task_losses) if task_losses else 0.0,
            "max_loss": np.max(task_losses) if task_losses else 0.0,
        }
    
    def descriptor(self, params: np.ndarray, metrics: Dict[str, float]) -> List[float]:
        """
        Compute behavioral descriptor.
        
        Descriptor dimensions:
        1. Complexity: model complexity/sparsity
        2. Generalization: consistency across tasks
        
        Args:
            params: Model parameters.
            metrics: Evaluation metrics.
            
        Returns:
            List of descriptor values in [0, 1].
        """
        # Complexity: based on parameter magnitude
        if isinstance(params, np.ndarray):
            param_norm = np.linalg.norm(params)
            complexity = min(1.0, param_norm / 10.0)
        else:
            complexity = 0.5
        
        # Generalization: based on consistency (low variance)
        std_loss = metrics.get("std_loss", 1.0)
        generalization = 1.0 / (1.0 + std_loss)
        
        return [complexity, generalization]
    
    def mutate(self, params: np.ndarray, magnitude: float) -> np.ndarray:
        """
        Mutate model parameters.
        
        Args:
            params: Original parameters.
            magnitude: Mutation magnitude.
            
        Returns:
            Mutated parameters.
        """
        expected_size = self.input_dim * self.output_dim + self.output_dim
        
        if not isinstance(params, np.ndarray):
            params = np.random.randn(expected_size) * 0.1
        
        noise = np.random.randn(*params.shape) * magnitude
        return params + noise
    
    def crossover(self, params_a: np.ndarray, params_b: np.ndarray) -> np.ndarray:
        """
        Crossover two parameter sets.
        
        Args:
            params_a: First parent.
            params_b: Second parent.
            
        Returns:
            Child parameters.
        """
        expected_size = self.input_dim * self.output_dim + self.output_dim
        
        if not isinstance(params_a, np.ndarray):
            params_a = np.random.randn(expected_size) * 0.1
        if not isinstance(params_b, np.ndarray):
            params_b = np.random.randn(expected_size) * 0.1
        
        # Ensure same shape
        if params_a.shape != params_b.shape:
            min_size = min(params_a.size, params_b.size)
            params_a = params_a.flatten()[:min_size]
            params_b = params_b.flatten()[:min_size]
        
        # Blend crossover with golden ratio
        alpha = 0.618  # phi - 1
        child = alpha * params_a + (1 - alpha) * params_b
        
        return child
    
    def task_sampler(self, n: int, difficulty: float) -> List[Dict[str, Any]]:
        """
        Sample supervised learning tasks.
        
        Args:
            n: Number of tasks.
            difficulty: Difficulty level in [0, 1].
            
        Returns:
            List of task specifications.
        """
        expected_size = self.input_dim * self.output_dim + self.output_dim
        tasks = []
        
        for i in range(n):
            # Noise increases with difficulty
            noise = 0.1 + difficulty * 0.9
            
            # Target weights
            target = np.random.randn(expected_size) * 0.5
            
            tasks.append({
                "id": i,
                "type": self.task_type,
                "noise": noise,
                "target": target.tolist(),
                "difficulty": difficulty,
            })
        
        return tasks
