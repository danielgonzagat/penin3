"""
NEURAL EVOLUTION CORE - Extraído de projetos IA³ antigos
Algoritmos de neuro-evolução REAIS e FUNCIONAIS

Fontes:
- agi-alpha-real/meta_evolver.py
- IA3_REAL/autoevolution_ia3.py  
- real_intelligence_system/neural_farm.py
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import copy
import json
import time

logger = logging.getLogger(__name__)

class NeuralGenome:
    """
    Representação genética de uma rede neural
    Inspirado em agi-alpha-real/meta_evolver.py (Genome class)
    """
    
    def __init__(self, layers: Tuple[int, ...] = (32, 64, 32),
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 dropout: float = 0.0):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.fitness = 0.0
        self.age = 0
    
    def mutate(self, mutation_rate: float = 0.3) -> 'NeuralGenome':
        """Mutate genome with controlled randomness"""
        new_genome = copy.deepcopy(self)
        
        # Mutate architecture
        if np.random.random() < mutation_rate:
            layers = list(new_genome.layers)
            
            # Add/remove layer
            if np.random.random() < 0.3 and len(layers) < 5:
                idx = np.random.randint(0, len(layers) + 1)
                layers.insert(idx, np.random.choice([16, 32, 64, 128]))
            elif len(layers) > 1 and np.random.random() < 0.2:
                idx = np.random.randint(0, len(layers))
                layers.pop(idx)
            
            # Change layer size
            if layers:
                idx = np.random.randint(0, len(layers))
                delta = np.random.randint(-16, 17)
                layers[idx] = max(8, min(256, layers[idx] + delta))
            
            new_genome.layers = tuple(layers)
        
        # Mutate hyperparameters
        if np.random.random() < mutation_rate:
            new_genome.learning_rate *= np.random.uniform(0.5, 2.0)
            new_genome.learning_rate = max(1e-5, min(0.1, new_genome.learning_rate))
        
        if np.random.random() < mutation_rate * 0.5:
            new_genome.dropout += np.random.uniform(-0.1, 0.1)
            new_genome.dropout = max(0.0, min(0.5, new_genome.dropout))
        
        if np.random.random() < mutation_rate * 0.3:
            new_genome.activation = np.random.choice(['relu', 'tanh', 'sigmoid', 'gelu'])
        
        return new_genome
    
    def crossover(self, other: 'NeuralGenome') -> 'NeuralGenome':
        """Crossover with another genome"""
        child = NeuralGenome()
        
        # Mix layers
        min_len = min(len(self.layers), len(other.layers))
        layers = []
        for i in range(min_len):
            layers.append(self.layers[i] if np.random.random() < 0.5 else other.layers[i])
        
        child.layers = tuple(layers)
        
        # Mix hyperparameters
        child.learning_rate = (self.learning_rate + other.learning_rate) / 2
        child.dropout = (self.dropout + other.dropout) / 2
        child.activation = self.activation if np.random.random() < 0.5 else other.activation
        
        return child
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (JSON-safe)"""
        return {
            'layers': [int(l) for l in self.layers],
            'activation': str(self.activation),
            'learning_rate': float(self.learning_rate),
            'dropout': float(self.dropout),
            'fitness': float(self.fitness),
            'age': int(self.age)
        }


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for neural architectures
    Extraído e aprimorado de múltiplos projetos IA³
    """
    
    def __init__(self, population_size: int = 20,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.5,
                 elite_ratio: float = 0.2,
                 checkpoint_dir: Path = None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.checkpoint_dir = checkpoint_dir or Path('data/evolution')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.population: List[NeuralGenome] = []
        self.generation = 0
        self.best_genome = None
        self.best_fitness = 0.0
        
        self._initialize_population()
        
        logger.info(f"🧬 Evolutionary Optimizer initialized")
        logger.info(f"   Population: {population_size}, Mutation: {mutation_rate}")
    
    def _initialize_population(self):
        """Initialize random population"""
        for _ in range(self.population_size):
            genome = NeuralGenome(
                layers=tuple(np.random.choice([16, 32, 64], size=np.random.randint(1, 4))),
                activation=np.random.choice(['relu', 'tanh', 'gelu']),
                learning_rate=10 ** np.random.uniform(-5, -2),
                dropout=np.random.uniform(0.0, 0.3)
            )
            self.population.append(genome)
    
    def evolve_generation(self, fitness_fn: callable) -> Dict[str, Any]:
        """
        Evolve one generation
        
        Args:
            fitness_fn: Function that takes a genome and returns fitness score
        
        Returns:
            Generation statistics
        """
        # Evaluate fitness
        for genome in self.population:
            genome.fitness = fitness_fn(genome)
            genome.age += 1
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Update best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_genome = copy.deepcopy(self.population[0])
        
        # Selection (elite + offspring)
        num_elite = int(self.population_size * self.elite_ratio)
        elite = self.population[:num_elite]
        
        # Create offspring
        offspring = []
        while len(offspring) < self.population_size - num_elite:
            # Selection
            parent1 = self._tournament_select()
            
            if np.random.random() < self.crossover_rate:
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = child.mutate(self.mutation_rate)
            
            offspring.append(child)
        
        # New population
        self.population = elite + offspring
        self.generation += 1
        
        # Stats
        fitnesses = [g.fitness for g in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'best_genome': self.best_genome.to_dict() if self.best_genome else None
        }
        
        # DETAILED LOGGING
        logger.info(f"🧬 Gen {self.generation}: "
                   f"best={self.best_fitness:.4f}, "
                   f"avg={stats['avg_fitness']:.4f}, "
                   f"std={stats['std_fitness']:.4f}, "
                   f"best_arch={self.best_genome.layers if self.best_genome else 'None'}")
        
        return stats
    
    def _tournament_select(self, tournament_size: int = 3) -> NeuralGenome:
        """Tournament selection"""
        tournament = np.random.choice(self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness)
    
    def save_checkpoint(self):
        """Save evolution checkpoint"""
        checkpoint = {
            'generation': self.generation,
            'population': [g.to_dict() for g in self.population],
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_fitness': self.best_fitness
        }
        
        path = self.checkpoint_dir / f'evolution_gen_{self.generation}.json'
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }
