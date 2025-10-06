"""
Open-Ended Evolution for Darwinacci
Evolution without fixed fitness - auto-generation of tasks
"""

import random
import logging
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenEndedTask:
    """Represents an open-ended task"""
    task_id: str
    task_type: str
    difficulty: float  # 0.0-1.0
    params: Dict[str, Any]
    objective: str
    
    def __hash__(self):
        return hash(self.task_id)


class OpenEndedEvolution:
    """
    Evolution without fixed fitness function
    Automatically generates new tasks as agents master existing ones
    
    Inspired by:
    - POET (Paired Open-Ended Trailblazer)
    - Quality Diversity
    - Minimal Criterion Coevolution
    """
    
    def __init__(self, base_env: str = 'CartPole-v1', seed: int = 42):
        self.base_env = base_env
        self.rng = random.Random(seed)
        self.task_archive: List[OpenEndedTask] = []
        self.mastered_tasks: List[OpenEndedTask] = []
        self.complexity_level = 1.0
        self.task_id_counter = 0
        
        # Initialize with base task
        self._add_base_tasks()
        
        logger.info(f"🌱 OpenEndedEvolution initialized: base_env={base_env}")
    
    def _add_base_tasks(self):
        """Add initial base tasks"""
        base_tasks = [
            OpenEndedTask(
                task_id='base_cartpole',
                task_type='rl_environment',
                difficulty=0.5,
                params={'env': 'CartPole-v1', 'gravity': 9.8, 'pole_length': 0.5},
                objective='survive_500_steps'
            ),
            OpenEndedTask(
                task_id='base_classification',
                task_type='classification',
                difficulty=0.3,
                params={'n_classes': 3, 'input_dim': 100},
                objective='accuracy_90'
            ),
            OpenEndedTask(
                task_id='base_regression',
                task_type='regression',
                difficulty=0.4,
                params={'function': 'linear', 'input_dim': 5},
                objective='mse_below_0.1'
            )
        ]
        
        self.task_archive.extend(base_tasks)
    
    def generate_new_task(self) -> OpenEndedTask:
        """
        Generate a new task automatically
        
        Strategy:
        1. If complexity < 2.0: Vary parameters of existing tasks
        2. If complexity < 5.0: Combine multiple tasks
        3. If complexity >= 5.0: Create meta-tasks (tasks about tasks)
        """
        self.task_id_counter += 1
        
        if self.complexity_level < 2.0:
            # Variation strategy
            return self._vary_existing_task()
        
        elif self.complexity_level < 5.0:
            # Combination strategy
            return self._combine_tasks()
        
        else:
            # Meta-task strategy
            return self._create_meta_task()
    
    def _vary_existing_task(self) -> OpenEndedTask:
        """Vary parameters of an existing task"""
        if not self.task_archive:
            self._add_base_tasks()
        
        base_task = self.rng.choice(self.task_archive)
        
        if base_task.task_type == 'rl_environment':
            # Vary environment parameters
            new_params = dict(base_task.params)
            
            if 'gravity' in new_params:
                new_params['gravity'] = self.rng.uniform(5.0, 15.0)
            
            if 'pole_length' in new_params:
                new_params['pole_length'] = self.rng.uniform(0.3, 1.0)
            
            new_difficulty = min(1.0, base_task.difficulty + self.rng.uniform(-0.1, 0.2))
            
            return OpenEndedTask(
                task_id=f'rl_var_{self.task_id_counter}',
                task_type='rl_environment',
                difficulty=new_difficulty,
                params=new_params,
                objective=base_task.objective
            )
        
        elif base_task.task_type == 'classification':
            new_params = dict(base_task.params)
            new_params['n_classes'] = self.rng.randint(2, 15)
            new_params['input_dim'] = self.rng.choice([50, 100, 200, 400])
            new_difficulty = min(1.0, new_params['n_classes'] / 15.0)
            
            return OpenEndedTask(
                task_id=f'clf_var_{self.task_id_counter}',
                task_type='classification',
                difficulty=new_difficulty,
                params=new_params,
                objective='accuracy_85'
            )
        
        elif base_task.task_type == 'regression':
            new_params = dict(base_task.params)
            new_params['function'] = self.rng.choice(['linear', 'quadratic', 'sinusoidal', 'exponential'])
            new_params['input_dim'] = self.rng.randint(1, 20)
            
            return OpenEndedTask(
                task_id=f'reg_var_{self.task_id_counter}',
                task_type='regression',
                difficulty=0.5,
                params=new_params,
                objective='mse_below_0.05'
            )
        
        # Fallback: return base task
        return base_task
    
    def _combine_tasks(self) -> OpenEndedTask:
        """Combine multiple tasks into a multi-objective task"""
        if len(self.task_archive) < 2:
            return self._vary_existing_task()
        
        tasks = self.rng.sample(self.task_archive, min(3, len(self.task_archive)))
        
        combined_objective = ' AND '.join([t.objective for t in tasks])
        combined_difficulty = sum(t.difficulty for t in tasks) / len(tasks)
        
        return OpenEndedTask(
            task_id=f'combined_{self.task_id_counter}',
            task_type='multi_objective',
            difficulty=min(1.0, combined_difficulty * 1.2),
            params={'subtasks': [t.task_id for t in tasks]},
            objective=combined_objective
        )
    
    def _create_meta_task(self) -> OpenEndedTask:
        """Create a meta-task (task about learning tasks)"""
        # Sample some mastered tasks
        if len(self.mastered_tasks) < 3:
            return self._combine_tasks()
        
        reference_tasks = self.rng.sample(self.mastered_tasks, min(5, len(self.mastered_tasks)))
        
        return OpenEndedTask(
            task_id=f'meta_{self.task_id_counter}',
            task_type='meta_learning',
            difficulty=0.9,
            params={
                'reference_tasks': [t.task_id for t in reference_tasks],
                'requirement': 'learn_new_task_in_10_episodes'
            },
            objective='fast_adaptation'
        )
    
    def evaluate_on_task(self, genome: Dict[str, Any], task: OpenEndedTask, 
                         eval_fn: Optional[Callable] = None) -> float:
        """
        Evaluate genome on a specific task
        
        Args:
            genome: Agent genome (hyperparameters)
            task: Task to evaluate on
            eval_fn: Optional external evaluation function
        
        Returns:
            fitness: 0.0-1.0
        """
        if eval_fn:
            try:
                result = eval_fn(genome, task)
                return float(result.get('fitness', 0.0))
            except Exception as e:
                logger.warning(f"External eval failed: {e}")
                return 0.0
        
        # Simple evaluation based on task type
        if task.task_type == 'rl_environment':
            # Simulate: higher learning rate + more neurons = better for simple envs
            lr = genome.get('learning_rate', genome.get('lr', 1e-3))
            neurons = genome.get('hidden_size', genome.get('neurons', 64))
            
            # Simple heuristic fitness
            fitness = min(1.0, (lr * 1000 * neurons) / 100.0)
            return fitness
        
        elif task.task_type == 'classification':
            # More neurons = better for classification
            neurons = genome.get('hidden_size', genome.get('neurons', 64))
            fitness = min(1.0, neurons / 200.0)
            return fitness
        
        elif task.task_type == 'regression':
            # Lower learning rate = better for regression
            lr = genome.get('learning_rate', genome.get('lr', 1e-3))
            fitness = max(0.0, 1.0 - lr * 500)
            return fitness
        
        return 0.5  # Default
    
    def archive_if_mastered(self, genome: Dict[str, Any], task: OpenEndedTask, fitness: float) -> bool:
        """
        Archive task if mastered and increase complexity
        
        Args:
            genome: Agent genome
            task: Task attempted
            fitness: Achieved fitness
        
        Returns:
            True if task was mastered and archived
        """
        mastery_threshold = 0.9
        
        if fitness >= mastery_threshold:
            # Task mastered!
            if task not in self.mastered_tasks:
                self.mastered_tasks.append(task)
                logger.info(f"🏆 Task mastered: {task.task_id} (fitness={fitness:.3f})")
                
                # Increase complexity
                self.complexity_level += 0.1
                logger.info(f"   Complexity: {self.complexity_level - 0.1:.1f} → {self.complexity_level:.1f}")
                
                return True
        
        return False
    
    def get_current_task(self) -> OpenEndedTask:
        """Get current task to work on"""
        # Return a random unmastered task, or generate new one
        unmastered = [t for t in self.task_archive if t not in self.mastered_tasks]
        
        if unmastered and self.rng.random() < 0.7:  # 70% work on existing
            return self.rng.choice(unmastered)
        else:  # 30% generate new
            new_task = self.generate_new_task()
            self.task_archive.append(new_task)
            return new_task
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get open-ended evolution statistics"""
        return {
            'total_tasks': len(self.task_archive),
            'mastered_tasks': len(self.mastered_tasks),
            'complexity_level': self.complexity_level,
            'task_types': self._count_task_types(),
            'mastery_rate': len(self.mastered_tasks) / max(1, len(self.task_archive))
        }
    
    def _count_task_types(self) -> Dict[str, int]:
        """Count tasks by type"""
        counts = {}
        for task in self.task_archive:
            counts[task.task_type] = counts.get(task.task_type, 0) + 1
        return counts


# Integration with Darwinacci Engine
def integrate_open_ended(darwinacci_engine, n_cycles: int = 100):
    """
    Integrate open-ended evolution with Darwinacci
    
    Args:
        darwinacci_engine: DarwinacciEngine instance
        n_cycles: Number of open-ended cycles
    """
    oee = OpenEndedEvolution()
    
    for cycle in range(n_cycles):
        # Get current task
        current_task = oee.get_current_task()
        logger.info(f"📋 Cycle {cycle}: Working on {current_task.task_id} (difficulty={current_task.difficulty:.2f})")
        
        # Override fitness function to use current task
        original_eval = darwinacci_engine.eval_fn
        
        def task_based_eval(genome, rng):
            """Evaluate genome on current open-ended task"""
            fitness = oee.evaluate_on_task(genome, current_task, eval_fn=None)
            return {
                'objective': fitness,
                'task_id': current_task.task_id
            }
        
        darwinacci_engine.eval_fn = task_based_eval
        
        # Run one Darwinacci generation
        stats = darwinacci_engine.run(max_cycles=1)
        
        # Check if task was mastered
        best_fitness = stats.get('best_fitness', 0.0)
        best_genome = stats.get('best_genome', {})
        
        oee.archive_if_mastered(best_genome, current_task, best_fitness)
        
        # Restore original eval
        darwinacci_engine.eval_fn = original_eval
        
        # Print stats
        if cycle % 10 == 0:
            stats = oee.get_statistics()
            logger.info(f"   📊 OEE Stats: {stats}")
    
    logger.info(f"🎯 Open-ended evolution complete: {oee.get_statistics()}")
    return oee


if __name__ == "__main__":
    # Test open-ended evolution
    oee = OpenEndedEvolution()
    
    print("🌱 Initial tasks:")
    for task in oee.task_archive:
        print(f"   - {task.task_id}: {task.task_type} (difficulty={task.difficulty:.2f})")
    
    print("\n🎲 Generating new tasks...")
    for i in range(10):
        new_task = oee.generate_new_task()
        print(f"   - {new_task.task_id}: {new_task.task_type} (difficulty={new_task.difficulty:.2f})")
        oee.task_archive.append(new_task)
        
        # Simulate mastery
        if i % 3 == 0:
            oee.mastered_tasks.append(new_task)
            oee.complexity_level += 0.1
    
    print("\n📊 Final statistics:")
    stats = oee.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")