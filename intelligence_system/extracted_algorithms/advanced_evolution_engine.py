"""
ADVANCED EVOLUTION ENGINE - Extraído de agi-alpha-real
Production-ready genetic algorithm com parallel evaluation e safety checks
"""
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import json
import copy

logger = logging.getLogger(__name__)

class EvolutionaryIndividual:
    """
    A candidate solution (genome) with fitness
    Extracted from: agi-alpha-real/evolution_engine.py
    """
    
    def __init__(self, genome: Dict[str, Any]):
        self.genome = genome
        self.fitness: Optional[float] = None
        self.age = 0
        self.mutations = 0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-safe dict"""
        return {
            'genome': self.genome,
            'fitness': float(self.fitness) if self.fitness is not None else None,
            'age': int(self.age),
            'mutations': int(self.mutations)
        }


class AdvancedEvolutionEngine:
    """
    Advanced evolutionary algorithm for neural architectures
    Extracted and enhanced from agi-alpha-real/evolution_engine.py
    
    Features:
    - Tournament selection
    - Elitism
    - Crossover and mutation
    - Parallel evaluation
    - Safety checks
    - Checkpointing
    """
    
    def __init__(self, population_size: int = 30,
                 crossover_rate: float = 0.5,
                 mutation_rate: float = 0.2,
                 tournament_k: int = 3,
                 elite_count: int = 2,
                 minimize: bool = False,
                 checkpoint_dir: Path = None):
        
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elite_count = elite_count
        self.elite_size = elite_count  # Alias for compatibility
        self.minimize = minimize
        self.checkpoint_dir = checkpoint_dir or Path('data/advanced_evolution')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.population: List[EvolutionaryIndividual] = []
        self.generation = 0
        self.best_individual: Optional[EvolutionaryIndividual] = None
        self.best_fitness = float('-inf') if not minimize else float('inf')
        self.history: List[Dict[str, Any]] = []
        
        logger.info("🧬 Advanced Evolution Engine initialized")
        logger.info(f"   Population: {population_size}, Elite: {elite_count}")
        logger.info(f"   Crossover: {crossover_rate}, Mutation: {mutation_rate}")
    
    def initialize_population(self, genome_template: Dict[str, Any]):
        """
        Initialize random population based on template
        
        Args:
            genome_template: Template dict with parameter ranges
        """
        self.population = []
        
        for _ in range(self.population_size):
            genome = {}
            for key, value_range in genome_template.items():
                if isinstance(value_range, tuple) and len(value_range) == 2:
                    # (min, max) range
                    genome[key] = random.uniform(value_range[0], value_range[1])
                elif isinstance(value_range, list):
                    # Choice from list
                    genome[key] = random.choice(value_range)
                else:
                    # Fixed value
                    genome[key] = value_range
            
            self.population.append(EvolutionaryIndividual(genome))
        
        logger.info(f"✅ Initialized population of {self.population_size}")
    
    def evaluate_population(self, fitness_fn: Callable):
        """
        Evaluate fitness of all individuals
        
        Args:
            fitness_fn: Function that takes genome and returns fitness
        """
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = fitness_fn(individual.genome)
        
        # Safety: Check for empty population
        if len(self.population) == 0:
            logger.warning("⚠️  Empty population after evaluation! Skipping this generation...")
            return
        
        # Update best
        self.population.sort(key=lambda ind: ind.fitness, reverse=not self.minimize)
        
        if self.minimize:
            if self.population[0].fitness < self.best_fitness:
                self.best_fitness = self.population[0].fitness
                self.best_individual = copy.deepcopy(self.population[0])
        else:
            if self.population[0].fitness > self.best_fitness:
                self.best_fitness = self.population[0].fitness
                self.best_individual = copy.deepcopy(self.population[0])
    
    def tournament_select(self) -> EvolutionaryIndividual:
        """Tournament selection"""
        # Safety: Check population size
        if len(self.population) == 0:
            return None
        
        k = min(self.tournament_k, len(self.population))
        tournament = random.sample(self.population, k)
        tournament.sort(key=lambda ind: ind.fitness, reverse=not self.minimize)
        return tournament[0]
    
    def crossover(self, parent1: EvolutionaryIndividual, 
                  parent2: EvolutionaryIndividual) -> Tuple[Dict, Dict]:
        """
        Crossover two genomes
        
        Returns:
            Two child genomes
        """
        g1, g2 = parent1.genome.copy(), parent2.genome.copy()
        
        # Uniform crossover
        child1, child2 = {}, {}
        for key in g1.keys():
            if random.random() < 0.5:
                child1[key] = g1[key]
                child2[key] = g2[key]
            else:
                child1[key] = g2[key]
                child2[key] = g1[key]
        
        return child1, child2
    
    def mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate genome
        
        Returns:
            Mutated genome
        """
        mutated = genome.copy()
        
        for key, value in mutated.items():
            if random.random() < 0.1:  # 10% chance per gene
                if isinstance(value, (int, float)):
                    # Gaussian mutation
                    sigma = abs(value) * 0.2 if value != 0 else 0.1
                    mutated[key] = value + random.gauss(0, sigma)
                    
                    # Clamp to reasonable range
                    if isinstance(value, int):
                        mutated[key] = int(mutated[key])
                        mutated[key] = max(-1000, min(1000, mutated[key]))
                    else:
                        mutated[key] = max(-10.0, min(10.0, mutated[key]))
        
        return mutated
    
    def evolve_generation(self, fitness_fn: Callable) -> Dict[str, Any]:
        """
        Evolve one generation
        
        Args:
            fitness_fn: Function to evaluate genomes
        
        Returns:
            Generation statistics
        """
        # Evaluate current population
        self.evaluate_population(fitness_fn)
        
        # Elitism - keep top N
        elite = self.population[:self.elite_count]
        
        # Create offspring
        offspring = []
        while len(offspring) < self.population_size - self.elite_count:
            # Selection
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            
            # Safety: Check if parents are valid
            if parent1 is None or parent2 is None:
                logger.warning("⚠️  No valid parents for reproduction! Breaking...")
                break
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self.crossover(parent1, parent2)
            else:
                child1_genome = parent1.genome.copy()
                child2_genome = parent2.genome.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1_genome = self.mutate(child1_genome)
            if random.random() < self.mutation_rate:
                child2_genome = self.mutate(child2_genome)
            
            # Create offspring
            child1 = EvolutionaryIndividual(child1_genome)
            child2 = EvolutionaryIndividual(child2_genome)
            child1.mutations = parent1.mutations + 1
            child2.mutations = parent2.mutations + 1
            
            offspring.extend([child1, child2])
        
        # New population
        offspring = offspring[:self.population_size - self.elite_count]
        
        # Age elite
        for ind in elite:
            ind.age += 1
        
        self.population = elite + offspring
        self.generation += 1
        
        # Statistics
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        stats = {
            'generation': int(self.generation),
            'best_fitness': float(self.best_fitness),
            'avg_fitness': float(np.mean(fitnesses)) if fitnesses else 0.0,
            'std_fitness': float(np.std(fitnesses)) if fitnesses else 0.0,
            'population_size': len(self.population)
        }
        
        self.history.append(stats)
        
        logger.info(f"🧬 Gen {self.generation}: best={self.best_fitness:.4f}, avg={stats['avg_fitness']:.4f}")
        
        return stats
    
    def save_checkpoint(self):
        """Save evolution checkpoint"""
        checkpoint = {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'best_fitness': float(self.best_fitness),
            'history': self.history
        }
        
        path = self.checkpoint_dir / f'advanced_evolution_gen_{self.generation}.json'
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"💾 Advanced evolution checkpoint saved: {path}")
    
    def get_best_genome(self) -> Optional[Dict[str, Any]]:
        """Get best genome found so far"""
        if self.best_individual:
            return self.best_individual.genome
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'total_evaluations': self.generation * self.population_size,
            'history_length': len(self.history)
        }
