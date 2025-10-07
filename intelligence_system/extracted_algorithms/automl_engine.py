"""
AutoML Engine - Extracted from Auto-PyTorch concepts
Enables Neural Architecture Search and Hyperparameter Optimization

Key concepts extracted:
- Neural Architecture Search (NAS)
- Hyperparameter Optimization (HPO)
- Multi-fidelity optimization (simplified)
- Ensemble learning
- Pipeline optimization

Clean implementation - no heavy dependencies
Focused on core AutoML capabilities
"""

import logging
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of neural network layers"""
    LINEAR = "linear"
    CONV = "conv"
    LSTM = "lstm"
    DROPOUT = "dropout"
    BATCHNORM = "batchnorm"
    ACTIVATION = "activation"


@dataclass
class ArchitectureConfig:
    """Configuration for a neural architecture"""
    layers: List[Dict[str, Any]]
    input_size: int
    output_size: int
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batchnorm: bool = True
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'layers': self.layers,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batchnorm': self.use_batchnorm,
            'score': self.score
        }


class ArchitectureSpace:
    """
    Defines the search space for neural architectures
    Inspired by Auto-PyTorch's architecture search space
    """
    
    def __init__(self, input_size: int, output_size: int, task: str = "classification"):
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        
        # Search space hyperparameters
        self.hidden_sizes = [32, 64, 128, 256, 512]
        self.num_layers_range = (1, 5)
        self.dropout_range = (0.0, 0.5)
        self.activations = ["relu", "tanh", "elu"]
        
    def sample_architecture(self) -> ArchitectureConfig:
        """
        Sample a random architecture from the space
        
        Returns:
            ArchitectureConfig
        """
        num_layers = random.randint(*self.num_layers_range)
        
        layers = []
        current_size = self.input_size
        
        for i in range(num_layers):
            # Sample hidden size
            hidden_size = random.choice(self.hidden_sizes)
            
            # Add linear layer
            layers.append({
                'type': LayerType.LINEAR.value,
                'in_features': current_size,
                'out_features': hidden_size
            })
            
            # Add batch norm (optional)
            use_bn = random.choice([True, False])
            if use_bn:
                layers.append({
                    'type': LayerType.BATCHNORM.value,
                    'num_features': hidden_size
                })
            
            # Add activation
            activation = random.choice(self.activations)
            layers.append({
                'type': LayerType.ACTIVATION.value,
                'activation': activation
            })
            
            # Add dropout
            dropout = random.uniform(*self.dropout_range)
            if dropout > 0.1:
                layers.append({
                    'type': LayerType.DROPOUT.value,
                    'p': dropout
                })
            
            current_size = hidden_size
        
        # Output layer
        layers.append({
            'type': LayerType.LINEAR.value,
            'in_features': current_size,
            'out_features': self.output_size
        })
        
        # Create config
        config = ArchitectureConfig(
            layers=layers,
            input_size=self.input_size,
            output_size=self.output_size,
            activation=random.choice(self.activations),
            dropout_rate=random.uniform(*self.dropout_range),
            use_batchnorm=random.choice([True, False])
        )
        
        return config
    
    def mutate_architecture(self, config: ArchitectureConfig, mutation_rate: float = 0.3) -> ArchitectureConfig:
        """
        Mutate an architecture
        
        Args:
            config: Base architecture
            mutation_rate: Probability of mutation
        
        Returns:
            Mutated architecture
        """
        if random.random() > mutation_rate:
            return config
        
        # Mutate by sampling a new one (simple strategy)
        new_config = self.sample_architecture()
        new_config.score = config.score  # Preserve score
        
        return new_config


class NeuralArchitectureSearch:
    """
    Neural Architecture Search engine
    Inspired by Auto-PyTorch's NAS approach
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        task: str = "classification",
        population_size: int = 10,
        n_iterations: int = 5
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        self.population_size = population_size
        self.n_iterations = n_iterations
        
        self.search_space = ArchitectureSpace(input_size, output_size, task)
        self.population: List[ArchitectureConfig] = []
        self.best_architecture: Optional[ArchitectureConfig] = None
        self.search_history: List[Dict] = []
        
    def initialize_population(self):
        """Initialize population with random architectures"""
        self.population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture()
            self.population.append(arch)
        
        logger.info(f"🧬 Initialized NAS population: {self.population_size} architectures")
    
    def evaluate_architecture(
        self,
        config: ArchitectureConfig,
        evaluation_fn: Callable[[ArchitectureConfig], float]
    ) -> float:
        """
        Evaluate an architecture
        
        Args:
            config: Architecture to evaluate
            evaluation_fn: Function that returns performance score
        
        Returns:
            Performance score
        """
        try:
            score = evaluation_fn(config)
            config.score = score
            return score
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return 0.0
    
    def search(
        self,
        evaluation_fn: Callable[[ArchitectureConfig], float],
        max_time: Optional[float] = None
    ) -> ArchitectureConfig:
        """
        Run NAS to find best architecture
        
        Args:
            evaluation_fn: Function to evaluate architectures
            max_time: Maximum search time (not implemented)
        
        Returns:
            Best architecture found
        """
        logger.info("🔍 Starting Neural Architecture Search...")
        
        # Initialize
        if not self.population:
            self.initialize_population()
        
        # Evolutionary search
        for iteration in range(self.n_iterations):
            logger.info(f"   Iteration {iteration + 1}/{self.n_iterations}")
            
            # Evaluate population
            for config in self.population:
                if config.score == 0.0:
                    self.evaluate_architecture(config, evaluation_fn)
            
            # Sort by score
            self.population.sort(key=lambda x: x.score, reverse=True)
            
            # Track best
            current_best = self.population[0]
            if self.best_architecture is None or current_best.score > self.best_architecture.score:
                self.best_architecture = current_best
                logger.info(f"   🏆 New best: {current_best.score:.4f}")
            
            # Record history
            self.search_history.append({
                'iteration': iteration,
                'best_score': current_best.score,
                'mean_score': np.mean([c.score for c in self.population])
            })
            
            # Evolve population (keep top 50%, generate new 50%)
            elite_size = self.population_size // 2
            elite = self.population[:elite_size]
            
            # Generate new population
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                # Mutate from elite
                parent = random.choice(elite)
                child = self.search_space.mutate_architecture(parent)
                child.score = 0.0  # Reset score
                new_population.append(child)
            
            self.population = new_population
        
        # Ensure best_architecture has non-zero score by evaluating if needed
        if self.best_architecture and self.best_architecture.score == 0.0:
            try:
                self.best_architecture.score = self.evaluate_architecture(self.best_architecture, evaluation_fn)
            except Exception:
                pass
        logger.info(f"✅ NAS complete! Best score: {self.best_architecture.score if self.best_architecture else 0.0:.4f}")
        return self.best_architecture
    
    def get_search_statistics(self) -> Dict:
        """Get NAS statistics"""
        if not self.search_history:
            return {}
        
        return {
            'iterations': len(self.search_history),
            'best_score': self.best_architecture.score if self.best_architecture else 0.0,
            'final_mean_score': self.search_history[-1]['mean_score'],
            'improvement': self.search_history[-1]['best_score'] - self.search_history[0]['best_score']
        }


class HyperparameterOptimizer:
    """
    Hyperparameter optimization
    Simplified version of SMAC-like optimization
    """
    
    def __init__(self):
        self.search_history: List[Dict] = []
        self.best_config: Optional[Dict] = None
        self.best_score: float = 0.0
        
    def suggest_hyperparameters(self, search_space: Dict) -> Dict:
        """
        Suggest hyperparameters
        
        Args:
            search_space: Dict defining ranges for each hyperparameter
        
        Returns:
            Sampled hyperparameters
        """
        config = {}
        for param, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                config[param] = random.randint(low, high)
            else:
                config[param] = random.uniform(low, high)
        
        return config
    
    def update(self, config: Dict, score: float):
        """
        Update optimizer with result
        
        Args:
            config: Hyperparameter configuration
            score: Performance score
        """
        self.search_history.append({
            'config': config,
            'score': score
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_config = config
            logger.info(f"🎯 New best hyperparameters: {score:.4f}")


class EnsembleBuilder:
    """
    Ensemble model builder
    Inspired by Auto-PyTorch's ensemble selection
    """
    
    def __init__(self, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.ensemble_models: List[Tuple[Any, float]] = []  # (model, weight)
        
    def add_model(self, model: Any, performance: float):
        """
        Add model to ensemble candidate pool
        
        Args:
            model: Model (or model config)
            performance: Performance score
        """
        self.ensemble_models.append((model, performance))
        
        # Sort by performance
        self.ensemble_models.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top N
        if len(self.ensemble_models) > self.ensemble_size * 2:
            self.ensemble_models = self.ensemble_models[:self.ensemble_size * 2]
        
        logger.info(f"📦 Added model to ensemble pool ({len(self.ensemble_models)} total)")
    
    def build_ensemble(self) -> List[Tuple[Any, float]]:
        """
        Build final ensemble with weights
        
        Returns:
            List of (model, weight) tuples
        """
        if not self.ensemble_models:
            return []
        
        # Select top K models
        selected = self.ensemble_models[:self.ensemble_size]
        
        # Compute weights (normalized performance)
        total_performance = sum(p for _, p in selected)
        ensemble = [(m, p / total_performance) for m, p in selected]
        
        logger.info(f"🎯 Built ensemble with {len(ensemble)} models")
        for i, (_, weight) in enumerate(ensemble):
            logger.info(f"   Model {i+1}: weight={weight:.3f}")
        
        return ensemble
    
    def predict_ensemble(self, ensemble: List[Tuple[Any, float]], x: Any) -> Any:
        """
        Make prediction with ensemble
        
        Args:
            ensemble: List of (model, weight)
            x: Input data
        
        Returns:
            Ensemble prediction
        """
        # This is a stub - real implementation would call each model
        predictions = []
        for model, weight in ensemble:
            # pred = model(x)  # Real prediction
            # predictions.append((pred, weight))
            pass
        
        # Weighted average (stub)
        return None


class AutoMLOrchestrator:
    """
    Main AutoML orchestrator
    Coordinates NAS, HPO, and ensemble building
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        task: str = "classification"
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        
        self.nas_engine = NeuralArchitectureSearch(
            input_size=input_size,
            output_size=output_size,
            task=task,
            population_size=10,
            n_iterations=5
        )
        
        self.hpo_engine = HyperparameterOptimizer()
        self.ensemble_builder = EnsembleBuilder(ensemble_size=5)
        
        self.active = False
        self.best_pipeline: Optional[Dict] = None
        
    def activate(self):
        """Activate AutoML capabilities"""
        self.active = True
        logger.info("🤖 AutoML engine ACTIVATED")
        logger.info("   NAS: ✅")
        logger.info("   HPO: ✅")
        logger.info("   Ensemble: ✅")
    
    # Backwards-compatibility wrapper expected by V7 system
    def search_architecture(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapper to perform a quick NAS and return a dict-like result
        compatible with legacy callers.
        """
        def mock_eval(config: ArchitectureConfig) -> float:
            # Simple surrogate: more layers slightly better, with randomness
            import random as _r
            num_linear = sum(1 for l in config.layers if l.get('type') == 'linear')
            return min(0.99, 0.7 + _r.random() * 0.2 + num_linear * 0.01)
        best = self.run_nas(mock_eval)
        return {'best_arch': best.to_dict() if best else None}
    
    def run_nas(
        self,
        evaluation_fn: Callable[[ArchitectureConfig], float]
    ) -> ArchitectureConfig:
        """
        Run Neural Architecture Search
        
        Args:
            evaluation_fn: Function to evaluate architectures
        
        Returns:
            Best architecture found
        """
        if not self.active:
            logger.warning("AutoML engine not active!")
            return None
        
        best_arch = self.nas_engine.search(evaluation_fn)
        return best_arch
    
    def optimize_hyperparameters(
        self,
        search_space: Dict,
        evaluation_fn: Callable[[Dict], float],
        n_trials: int = 10
    ) -> Dict:
        """
        Optimize hyperparameters
        
        Args:
            search_space: Dict defining ranges
            evaluation_fn: Function to evaluate configs
            n_trials: Number of trials
        
        Returns:
            Best hyperparameters
        """
        if not self.active:
            logger.warning("AutoML engine not active!")
            return {}
        
        logger.info(f"🔧 Starting HPO with {n_trials} trials...")
        
        for trial in range(n_trials):
            # Suggest config
            config = self.hpo_engine.suggest_hyperparameters(search_space)
            
            # Evaluate
            score = evaluation_fn(config)
            
            # Update
            self.hpo_engine.update(config, score)
            
            logger.info(f"   Trial {trial+1}/{n_trials}: {score:.4f}")
        
        logger.info(f"✅ HPO complete! Best: {self.hpo_engine.best_score:.4f}")
        return self.hpo_engine.best_config
    
    def build_ensemble(
        self,
        models: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        """
        Build ensemble from models
        
        Args:
            models: List of (model, performance) tuples
        
        Returns:
            Ensemble with weights
        """
        for model, performance in models:
            self.ensemble_builder.add_model(model, performance)
        
        ensemble = self.ensemble_builder.build_ensemble()
        return ensemble
    
    def get_status(self) -> Dict[str, Any]:
        """Get AutoML status"""
        nas_stats = self.nas_engine.get_search_statistics()
        
        return {
            'active': self.active,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'task': self.task,
            'nas_stats': nas_stats,
            'hpo_best_score': self.hpo_engine.best_score,
            'ensemble_size': len(self.ensemble_builder.ensemble_models)
        }


# Test function
def test_automl_engine():
    """Test the AutoML engine"""
    print("="*80)
    print("🧪 TESTING AUTOML ENGINE")
    print("="*80)
    
    # Initialize
    engine = AutoMLOrchestrator(
        input_size=784,  # MNIST
        output_size=10,
        task="classification"
    )
    engine.activate()
    
    # Mock evaluation function
    def mock_evaluate_arch(config: ArchitectureConfig) -> float:
        # Mock score based on number of layers
        num_linear = sum(1 for l in config.layers if l.get('type') == 'linear')
        score = 0.7 + random.random() * 0.2 + num_linear * 0.01
        return min(score, 0.99)
    
    # Test NAS
    print("\n🔍 Testing Neural Architecture Search:")
    best_arch = engine.run_nas(mock_evaluate_arch)
    print(f"   Best architecture score: {best_arch.score:.4f}")
    print(f"   Number of layers: {len(best_arch.layers)}")
    
    # Test HPO
    print("\n🔧 Testing Hyperparameter Optimization:")
    search_space = {
        'learning_rate': (0.0001, 0.01),
        'batch_size': (16, 128),
        'epochs': (5, 50)
    }
    
    def mock_evaluate_hpo(config: Dict) -> float:
        # Mock score
        return 0.8 + random.random() * 0.15
    
    best_hpo = engine.optimize_hyperparameters(search_space, mock_evaluate_hpo, n_trials=5)
    print(f"   Best hyperparameters: {best_hpo}")
    
    # Test ensemble
    print("\n📦 Testing Ensemble Building:")
    mock_models = [
        ("model_1", 0.85),
        ("model_2", 0.88),
        ("model_3", 0.82),
        ("model_4", 0.90),
        ("model_5", 0.87)
    ]
    ensemble = engine.build_ensemble(mock_models)
    
    # Get status
    print("\n📊 Engine Status:")
    status = engine.get_status()
    for key, value in status.items():
        if key != 'nas_stats':
            print(f"   {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ AUTOML ENGINE TEST COMPLETE")
    print("="*80)
    
    return engine


if __name__ == "__main__":
    # Run test
    engine = test_automl_engine()
