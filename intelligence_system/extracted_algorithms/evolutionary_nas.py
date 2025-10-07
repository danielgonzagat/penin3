"""
Evolutionary Neural Architecture Search (NAS)
Evolves neural network architectures via Darwinian evolution
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class NetworkArchitecture:
    """Represents a neural network architecture"""
    arch_id: str
    n_layers: int
    layer_sizes: List[int]
    activations: List[str]
    connections: Dict[str, Any]  # Skip connections, etc.
    normalization: str
    dropout: float
    fitness: float = 0.0
    
    def __hash__(self):
        return hash(self.arch_id)
    
    def to_dict(self) -> Dict:
        return {
            'arch_id': self.arch_id,
            'n_layers': self.n_layers,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'connections': self.connections,
            'normalization': self.normalization,
            'dropout': self.dropout,
            'fitness': self.fitness
        }


class EvolutionaryNAS:
    """
    Evolves neural network architectures via genetic algorithm
    
    Inspired by:
    - NEAT (NeuroEvolution of Augmenting Topologies)
    - DARTS (Differentiable Architecture Search)
    - AmoebaNet
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 input_dim: int = 784,
                 output_dim: int = 10,
                 seed: int = 42):
        self.population_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        self.population: List[NetworkArchitecture] = []
        self.generation = 0
        self.best_architecture: Optional[NetworkArchitecture] = None
        self.fitness_history: List[float] = []
        
        logger.info(f"🧬 EvolutionaryNAS initialized: pop_size={population_size}, input={input_dim}, output={output_dim}")
    
    def initialize_population(self):
        """Create initial population of diverse architectures"""
        logger.info("🌱 Initializing population...")
        
        for i in range(self.population_size):
            arch = self._random_architecture(arch_id=f"gen0_arch{i}")
            self.population.append(arch)
        
        logger.info(f"   ✅ Created {len(self.population)} architectures")
    
    def _random_architecture(self, arch_id: str) -> NetworkArchitecture:
        """Generate random architecture"""
        n_layers = self.rng.randint(2, 8)
        
        # Generate layer sizes (gradually decrease toward output)
        layer_sizes = []
        current_size = self.input_dim
        
        for i in range(n_layers):
            # Decrease size gradually
            next_size = self.rng.randint(32, min(512, current_size))
            layer_sizes.append(next_size)
            current_size = next_size
        
        # Ensure output layer
        layer_sizes.append(self.output_dim)
        
        # Random activations
        activation_choices = ['relu', 'tanh', 'gelu', 'sigmoid']
        activations = [self.rng.choice(activation_choices) for _ in range(n_layers)]
        
        # Skip connections (random)
        connections = {
            'skip_connections': [
                (i, i+2) for i in range(n_layers - 1)
                if self.rng.random() < 0.3  # 30% chance
            ]
        }
        
        # Normalization
        normalization = self.rng.choice(['batch_norm', 'layer_norm', 'none'])
        
        # Dropout
        dropout = self.rng.uniform(0.0, 0.5)
        
        return NetworkArchitecture(
            arch_id=arch_id,
            n_layers=n_layers,
            layer_sizes=layer_sizes,
            activations=activations,
            connections=connections,
            normalization=normalization,
            dropout=dropout
        )
    
    def build_model(self, architecture: NetworkArchitecture) -> nn.Module:
        """Build PyTorch model from architecture specification"""
        
        class DynamicNet(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.arch = arch
                self.layers = nn.ModuleList()
                
                # Build layers
                for i in range(len(arch.layer_sizes) - 1):
                    in_dim = arch.layer_sizes[i]
                    out_dim = arch.layer_sizes[i + 1]
                    
                    self.layers.append(nn.Linear(in_dim, out_dim))
            
            def forward(self, x):
                # Sequential forward
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    
                    # Apply activation
                    if i < len(self.arch.activations):
                        act_name = self.arch.activations[i]
                        if act_name == 'relu':
                            x = torch.relu(x)
                        elif act_name == 'tanh':
                            x = torch.tanh(x)
                        elif act_name == 'gelu':
                            x = torch.nn.functional.gelu(x)
                        elif act_name == 'sigmoid':
                            x = torch.sigmoid(x)
                    
                    # Apply dropout (during training)
                    if self.training and self.arch.dropout > 0:
                        x = torch.nn.functional.dropout(x, p=self.arch.dropout)
                
                return x
        
        return DynamicNet(architecture)
    
    def evaluate_architecture(self, architecture: NetworkArchitecture, 
                            train_data: Tuple, test_data: Tuple,
                            epochs: int = 5) -> float:
        """
        Train network with this architecture and measure performance
        
        Args:
            architecture: Architecture to evaluate
            train_data: (X_train, y_train)
            test_data: (X_test, y_test)
            epochs: Training epochs
        
        Returns:
            fitness: Test accuracy
        """
        try:
            # Build model
            model = self.build_model(architecture)
            
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            X_train, y_train = train_data
            X_test, y_test = test_data
            
            # Quick training
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test).float().mean().item()
            
            return accuracy
        
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return 0.0
    
    def evolve_generation(self, train_data: Tuple, test_data: Tuple) -> Dict[str, Any]:
        """
        Evolve one generation
        
        Returns:
            Statistics about this generation
        """
        logger.info(f"🧬 Evolving generation {self.generation}...")
        
        # 1. Evaluate all architectures
        logger.info(f"   📊 Evaluating {len(self.population)} architectures...")
        fitnesses = []
        
        for arch in self.population:
            fitness = self.evaluate_architecture(arch, train_data, test_data, epochs=3)
            arch.fitness = fitness
            fitnesses.append(fitness)
        
        avg_fitness = np.mean(fitnesses)
        best_fitness = np.max(fitnesses)
        
        logger.info(f"   📊 Gen {self.generation}: avg={avg_fitness:.3f}, best={best_fitness:.3f}")
        
        # Update best
        best_idx = np.argmax(fitnesses)
        self.best_architecture = self.population[best_idx]
        self.fitness_history.append(best_fitness)
        
        # 2. Selection: top 50%
        sorted_pop = sorted(zip(self.population, fitnesses), key=lambda x: x[1], reverse=True)
        survivors = [arch for arch, _ in sorted_pop[:self.population_size // 2]]
        
        logger.info(f"   🔪 Selected top {len(survivors)} architectures")
        
        # 3. Reproduction: crossover + mutation
        offspring = []
        while len(offspring) < self.population_size // 2:
            parent1 = self.rng.choice(survivors)
            parent2 = self.rng.choice(survivors)
            
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            
            offspring.append(child)
        
        # 4. New population
        self.population = survivors + offspring
        self.generation += 1
        
        return {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'best_arch': self.best_architecture.to_dict() if self.best_architecture else None
        }
    
    def _crossover(self, arch1: NetworkArchitecture, arch2: NetworkArchitecture) -> NetworkArchitecture:
        """Crossover two architectures"""
        child_id = f"gen{self.generation}_arch{self.rng.randint(0, 10000)}"
        
        # Mix parameters from both parents
        child = NetworkArchitecture(
            arch_id=child_id,
            n_layers=self.rng.choice([arch1.n_layers, arch2.n_layers]),
            layer_sizes=list(self.rng.choice([arch1.layer_sizes, arch2.layer_sizes])),
            activations=list(self.rng.choice([arch1.activations, arch2.activations])),
            connections=dict(self.rng.choice([arch1.connections, arch2.connections])),
            normalization=self.rng.choice([arch1.normalization, arch2.normalization]),
            dropout=self.rng.choice([arch1.dropout, arch2.dropout])
        )
        
        return child
    
    def _mutate(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Mutate architecture"""
        mutated = deepcopy(architecture)
        mutated.arch_id = f"{architecture.arch_id}_mut"
        
        # Mutate n_layers (30% chance)
        if self.rng.random() < 0.3:
            mutated.n_layers = max(2, min(10, mutated.n_layers + self.rng.randint(-1, 1)))
        
        # Mutate layer sizes (40% chance)
        if self.rng.random() < 0.4 and mutated.layer_sizes:
            idx = self.rng.randint(0, len(mutated.layer_sizes) - 1)
            mutated.layer_sizes[idx] = max(16, min(512, mutated.layer_sizes[idx] + self.rng.randint(-64, 64)))
        
        # Mutate activation (20% chance)
        if self.rng.random() < 0.2 and mutated.activations:
            idx = self.rng.randint(0, len(mutated.activations) - 1)
            mutated.activations[idx] = self.rng.choice(['relu', 'tanh', 'gelu', 'sigmoid'])
        
        # Mutate dropout (30% chance)
        if self.rng.random() < 0.3:
            mutated.dropout = max(0.0, min(0.5, mutated.dropout + self.rng.uniform(-0.1, 0.1)))
        
        # Mutate normalization (20% chance)
        if self.rng.random() < 0.2:
            mutated.normalization = self.rng.choice(['batch_norm', 'layer_norm', 'none'])
        
        return mutated
    
    def get_best_model(self) -> Optional[nn.Module]:
        """Get best model found so far"""
        if self.best_architecture is None:
            return None
        
        return self.build_model(self.best_architecture)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get NAS statistics"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_architecture.fitness if self.best_architecture else 0.0,
            'fitness_history': self.fitness_history,
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None
        }


# Integration helper for V7
def run_nas_for_v7(task: str = 'mnist', generations: int = 10) -> Dict[str, Any]:
    """
    Run NAS for V7 system
    
    Args:
        task: 'mnist' or 'cartpole'
        generations: Number of evolutionary generations
    
    Returns:
        Best architecture found
    """
    logger.info(f"🔬 Running NAS for {task} ({generations} generations)...")
    
    if task == 'mnist':
        # Load MNIST data (small subset for speed)
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('/root/mnist_data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('/root/mnist_data', train=False, transform=transform)
        
        # Use subset for speed
        train_subset = torch.utils.data.Subset(train_dataset, range(1000))
        test_subset = torch.utils.data.Subset(test_dataset, range(200))
        
        # Convert to tensors
        X_train = torch.stack([train_subset[i][0].view(-1) for i in range(len(train_subset))])
        y_train = torch.LongTensor([train_subset[i][1] for i in range(len(train_subset))])
        
        X_test = torch.stack([test_subset[i][0].view(-1) for i in range(len(test_subset))])
        y_test = torch.LongTensor([test_subset[i][1] for i in range(len(test_subset))])
        
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        
        # Initialize NAS
        nas = EvolutionaryNAS(
            population_size=20,
            input_dim=784,
            output_dim=10
        )
        
        # Initialize population
        nas.initialize_population()
        
        # Evolve
        for gen in range(generations):
            stats = nas.evolve_generation(train_data, test_data)
            logger.info(f"   Gen {gen}: best={stats['best_fitness']:.3f}")
        
        # Return best
        result = nas.get_statistics()
        logger.info(f"   🏆 Best architecture: {result['best_fitness']:.3f} accuracy")
        
        return result
    
    else:
        logger.warning(f"NAS for {task} not implemented yet")
        return {'error': f'NAS for {task} not implemented'}


if __name__ == "__main__":
    # Test evolutionary NAS
    print("🧬 Testing Evolutionary NAS...")
    
    # Create synthetic data for testing
    X_train = torch.randn(100, 784)
    y_train = torch.randint(0, 10, (100,))
    X_test = torch.randn(20, 784)
    y_test = torch.randint(0, 10, (20,))
    
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)
    
    # Initialize NAS
    nas = EvolutionaryNAS(population_size=10, input_dim=784, output_dim=10)
    nas.initialize_population()
    
    # Evolve for 5 generations
    print("\n🧬 Evolving for 5 generations...")
    for gen in range(5):
        stats = nas.evolve_generation(train_data, test_data)
        print(f"   Gen {gen}: avg={stats['avg_fitness']:.3f}, best={stats['best_fitness']:.3f}")
    
    # Get best
    print("\n🏆 Best architecture found:")
    best = nas.best_architecture
    if best:
        print(f"   ID: {best.arch_id}")
        print(f"   Layers: {best.n_layers}")
        print(f"   Sizes: {best.layer_sizes}")
        print(f"   Activations: {best.activations}")
        print(f"   Fitness: {best.fitness:.3f}")
    
    print("\n✅ NAS test complete")