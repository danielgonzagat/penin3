"""
Tool Pipeline Adapter: Text Processing Pipeline

A synthetic tool for text processing/transformation tasks.
The "parameters" control transformation rules.
"""

import numpy as np
from typing import Dict, List, Any
import re


class ToolPipelineAdapter:
    """
    Adapter for tool pipeline optimization.
    
    Task: Optimize a text processing pipeline.
    - Input: text string
    - Parameters: transformation weights/rules
    - Output: transformed text
    - Quality: similarity to target transformation
    
    This is a synthetic example showing how to wrap any tool/pipeline.
    
    Args:
        n_transformations: Number of available transformations.
    """
    
    def __init__(self, n_transformations: int = 8):
        self.n_transformations = n_transformations
        
        # Define synthetic transformations
        self.transformations = [
            self._transform_uppercase,
            self._transform_lowercase,
            self._transform_reverse,
            self._transform_remove_vowels,
            self._transform_double_consonants,
            self._transform_shuffle,
            self._transform_rot13,
            self._transform_leetspeak,
        ]
    
    def _transform_uppercase(self, text: str) -> str:
        return text.upper()
    
    def _transform_lowercase(self, text: str) -> str:
        return text.lower()
    
    def _transform_reverse(self, text: str) -> str:
        return text[::-1]
    
    def _transform_remove_vowels(self, text: str) -> str:
        return re.sub(r'[aeiouAEIOU]', '', text)
    
    def _transform_double_consonants(self, text: str) -> str:
        return re.sub(r'([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])', r'\1\1', text)
    
    def _transform_shuffle(self, text: str) -> str:
        chars = list(text)
        np.random.shuffle(chars)
        return ''.join(chars)
    
    def _transform_rot13(self, text: str) -> str:
        return ''.join(
            chr((ord(c) - 65 + 13) % 26 + 65) if c.isupper()
            else chr((ord(c) - 97 + 13) % 26 + 97) if c.islower()
            else c
            for c in text
        )
    
    def _transform_leetspeak(self, text: str) -> str:
        mapping = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
        return ''.join(mapping.get(c.lower(), c) for c in text)
    
    def _apply_pipeline(self, text: str, params: np.ndarray) -> str:
        """
        Apply transformation pipeline based on parameters.
        
        Args:
            text: Input text.
            params: Pipeline weights (which transformations to apply).
            
        Returns:
            Transformed text.
        """
        # Parameters control which transformations are applied
        # Use softmax to get probabilities
        if len(params) < self.n_transformations:
            params = np.random.randn(self.n_transformations)
        
        probs = np.exp(params[:self.n_transformations])
        probs = probs / probs.sum()
        
        # Select top transformations
        n_apply = max(1, min(3, int(np.sum(probs > 0.2))))
        top_indices = np.argsort(probs)[-n_apply:]
        
        result = text
        for idx in top_indices:
            if idx < len(self.transformations):
                try:
                    result = self.transformations[idx](result)
                except:
                    pass
        
        return result
    
    def evaluate(self, params: np.ndarray, tasks: List[Any]) -> Dict[str, float]:
        """
        Evaluate pipeline parameters on tasks.
        
        Args:
            params: Pipeline parameters.
            tasks: List of task specifications (input/target pairs).
            
        Returns:
            Dictionary with 'fitness' and other metrics.
        """
        if not isinstance(params, np.ndarray):
            params = np.random.randn(self.n_transformations)
        
        total_score = 0.0
        task_scores = []
        
        for task in tasks:
            if isinstance(task, dict):
                input_text = task.get("input", "hello world")
                target_text = task.get("target", "HELLO WORLD")
            else:
                input_text = "hello world"
                target_text = "HELLO WORLD"
            
            # Apply pipeline
            output = self._apply_pipeline(input_text, params)
            
            # Compute similarity (simple character overlap)
            score = self._text_similarity(output, target_text)
            
            task_scores.append(score)
            total_score += score
        
        mean_score = total_score / len(tasks) if tasks else 0.0
        
        return {
            "fitness": mean_score,
            "mean_score": mean_score,
            "std_score": np.std(task_scores) if task_scores else 0.0,
            "min_score": np.min(task_scores) if task_scores else 0.0,
            "max_score": np.max(task_scores) if task_scores else 0.0,
        }
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity score."""
        if not text1 or not text2:
            return 0.0
        
        # Character set overlap
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def descriptor(self, params: np.ndarray, metrics: Dict[str, float]) -> List[float]:
        """
        Compute behavioral descriptor.
        
        Descriptor dimensions:
        1. Diversity: how many transformations are used
        2. Consistency: variance of scores
        
        Args:
            params: Pipeline parameters.
            metrics: Evaluation metrics.
            
        Returns:
            List of descriptor values in [0, 1].
        """
        # Diversity: entropy of transformation weights
        if isinstance(params, np.ndarray) and len(params) >= self.n_transformations:
            probs = np.exp(params[:self.n_transformations])
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            diversity = min(1.0, entropy / 2.0)
        else:
            diversity = 0.5
        
        # Consistency
        std_score = metrics.get("std_score", 0.5)
        consistency = 1.0 / (1.0 + std_score)
        
        return [diversity, consistency]
    
    def mutate(self, params: np.ndarray, magnitude: float) -> np.ndarray:
        """
        Mutate pipeline parameters.
        
        Args:
            params: Original parameters.
            magnitude: Mutation magnitude.
            
        Returns:
            Mutated parameters.
        """
        if not isinstance(params, np.ndarray):
            params = np.random.randn(self.n_transformations)
        
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
        if not isinstance(params_a, np.ndarray):
            params_a = np.random.randn(self.n_transformations)
        if not isinstance(params_b, np.ndarray):
            params_b = np.random.randn(self.n_transformations)
        
        # Ensure same shape
        if params_a.shape != params_b.shape:
            min_size = min(params_a.size, params_b.size)
            params_a = params_a.flatten()[:min_size]
            params_b = params_b.flatten()[:min_size]
        
        # Single-point crossover
        point = np.random.randint(1, len(params_a))
        child = np.concatenate([params_a[:point], params_b[point:]])
        
        return child
    
    def task_sampler(self, n: int, difficulty: float) -> List[Dict[str, Any]]:
        """
        Sample tool pipeline tasks.
        
        Args:
            n: Number of tasks.
            difficulty: Difficulty level in [0, 1].
            
        Returns:
            List of task specifications.
        """
        tasks = []
        
        # Sample texts
        sample_texts = [
            "hello world",
            "fibonacci engine",
            "artificial intelligence",
            "quality diversity",
            "golden ratio",
            "evolution strategy",
        ]
        
        for i in range(n):
            input_text = np.random.choice(sample_texts)
            
            # Target is a transformation
            # Difficulty controls complexity
            if difficulty < 0.3:
                target = input_text.upper()
            elif difficulty < 0.6:
                target = input_text[::-1]
            else:
                target = re.sub(r'[aeiou]', '', input_text)
            
            tasks.append({
                "id": i,
                "type": "text_transform",
                "input": input_text,
                "target": target,
                "difficulty": difficulty,
            })
        
        return tasks
