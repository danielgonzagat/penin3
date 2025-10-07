"""
Curiosity-Driven Task Generation
Automatically generates new tasks based on agent's curiosity and current capabilities
"""

import random
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents agent's capability in a skill"""
    skill_name: str
    proficiency: float  # 0.0-1.0
    history: List[float]  # Performance history
    
    def is_improving(self) -> bool:
        """Check if capability is improving"""
        if len(self.history) < 3:
            return True
        recent = self.history[-3:]
        return recent[-1] > recent[0]
    
    def is_stagnant(self) -> bool:
        """Check if capability is stagnant"""
        if len(self.history) < 5:
            return False
        recent = self.history[-5:]
        std = np.std(recent)
        return std < 0.01  # Very little variation


class CuriosityTaskGenerator:
    """
    Generates tasks automatically based on curiosity
    
    Principles:
    1. Zone of Proximal Development: Tasks slightly harder than current level
    2. Curiosity-driven: Focus on skills that are improving or unexplored
    3. Anti-stagnation: Avoid skills that are stuck
    4. Novelty seeking: Prefer tasks agent hasn't seen before
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Track agent's capabilities
        self.capabilities: Dict[str, AgentCapability] = {}
        
        # Task history
        self.generated_tasks: List[Dict] = []
        self.task_performances: Dict[str, List[float]] = defaultdict(list)
        
        # Curiosity metrics
        self.curiosity_scores: Dict[str, float] = {}
        
        logger.info("🔍 CuriosityTaskGenerator initialized")
    
    def update_capability(self, skill: str, performance: float):
        """Update agent's capability in a skill"""
        if skill not in self.capabilities:
            self.capabilities[skill] = AgentCapability(
                skill_name=skill,
                proficiency=performance,
                history=[performance]
            )
        else:
            cap = self.capabilities[skill]
            cap.history.append(performance)
            cap.proficiency = np.mean(cap.history[-10:])  # Moving average
        
        # Update curiosity: reward improvement and exploration
        self._update_curiosity(skill)
    
    def _update_curiosity(self, skill: str):
        """Update curiosity score for a skill"""
        cap = self.capabilities[skill]
        
        # High curiosity for:
        # 1. Skills that are improving (learning is happening!)
        # 2. Skills that are unexplored (low proficiency, few attempts)
        # 3. Skills that show high variance (interesting dynamics)
        
        curiosity = 0.0
        
        # Factor 1: Improvement (gradient)
        if len(cap.history) >= 2:
            improvement = cap.history[-1] - cap.history[-2]
            curiosity += max(0, improvement) * 5.0
        
        # Factor 2: Unexplored (low proficiency + few attempts)
        exploration_bonus = (1.0 - cap.proficiency) * 0.3
        if len(cap.history) < 10:
            exploration_bonus *= 2.0  # Double bonus for unexplored
        curiosity += exploration_bonus
        
        # Factor 3: Variance (interesting != boring)
        if len(cap.history) >= 3:
            variance = np.std(cap.history[-10:])
            curiosity += variance * 2.0
        
        # Factor 4: Avoid stagnation (penalize stuck skills)
        if cap.is_stagnant():
            curiosity *= 0.5
        
        self.curiosity_scores[skill] = curiosity
    
    def generate_next_task(self, current_capabilities: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate next task based on curiosity
        
        Args:
            current_capabilities: Optional dict of {skill: proficiency}
        
        Returns:
            Task specification
        """
        # Update capabilities if provided
        if current_capabilities:
            for skill, perf in current_capabilities.items():
                self.update_capability(skill, perf)
        
        # Select skill to train based on curiosity
        target_skill = self._select_curious_skill()
        
        # Generate task for that skill
        task = self._generate_task_for_skill(target_skill)
        
        # Record generation
        self.generated_tasks.append({
            'task': task,
            'skill': target_skill,
            'curiosity': self.curiosity_scores.get(target_skill, 0.0)
        })
        
        logger.info(f"🎯 Generated task: {task['task_id']} for skill '{target_skill}' (curiosity={self.curiosity_scores.get(target_skill, 0):.3f})")
        
        return task
    
    def _select_curious_skill(self) -> str:
        """Select skill based on curiosity scores"""
        if not self.curiosity_scores:
            # No data yet - explore randomly
            default_skills = ['visual_recognition', 'sequential_reasoning', 'motor_control', 
                            'pattern_recognition', 'optimization', 'memory', 'planning']
            return self.rng.choice(default_skills)
        
        # Softmax selection: higher curiosity = higher probability
        skills = list(self.curiosity_scores.keys())
        curiosities = np.array([self.curiosity_scores[s] for s in skills])
        
        # Add epsilon for exploration
        epsilon = 0.1
        if self.rng.random() < epsilon:
            return self.rng.choice(skills)
        
        # Softmax
        exp_curiosities = np.exp(curiosities - np.max(curiosities))  # Numerical stability
        probabilities = exp_curiosities / np.sum(exp_curiosities)
        
        selected_idx = np.random.choice(len(skills), p=probabilities)
        return skills[selected_idx]
    
    def _generate_task_for_skill(self, skill: str) -> Dict[str, Any]:
        """Generate specific task for a skill"""
        # Get current proficiency in this skill
        proficiency = 0.5
        if skill in self.capabilities:
            proficiency = self.capabilities[skill].proficiency
        
        # Task difficulty: slightly above current level (ZPD)
        target_difficulty = min(1.0, proficiency + 0.15)
        
        task = {
            'task_id': f'{skill}_{len(self.generated_tasks)}',
            'skill': skill,
            'difficulty': target_difficulty,
            'type': self._skill_to_task_type(skill),
            'params': self._generate_task_params(skill, target_difficulty)
        }
        
        return task
    
    def _skill_to_task_type(self, skill: str) -> str:
        """Map skill to task type"""
        mapping = {
            'visual_recognition': 'image_classification',
            'sequential_reasoning': 'sequence_prediction',
            'motor_control': 'rl_environment',
            'pattern_recognition': 'pattern_completion',
            'optimization': 'optimization_problem',
            'memory': 'recall_task',
            'planning': 'planning_problem'
        }
        return mapping.get(skill, 'general_task')
    
    def _generate_task_params(self, skill: str, difficulty: float) -> Dict[str, Any]:
        """Generate parameters for task based on skill and difficulty"""
        if skill == 'visual_recognition':
            return {
                'n_classes': int(5 + difficulty * 10),
                'image_complexity': difficulty,
                'image_size': 28 if difficulty < 0.7 else 64
            }
        
        elif skill == 'sequential_reasoning':
            return {
                'sequence_length': int(5 + difficulty * 20),
                'pattern_complexity': difficulty
            }
        
        elif skill == 'motor_control':
            # Map to Gym environments by difficulty
            envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2']
            env_idx = int(min(len(envs) - 1, difficulty * len(envs)))
            return {
                'env_name': envs[env_idx],
                'target_reward': difficulty * 500
            }
        
        elif skill == 'pattern_recognition':
            return {
                'pattern_type': self.rng.choice(['arithmetic', 'geometric', 'fibonacci', 'prime']),
                'sequence_length': int(10 + difficulty * 20)
            }
        
        elif skill == 'optimization':
            return {
                'dimensions': int(2 + difficulty * 10),
                'function': self.rng.choice(['sphere', 'rastrigin', 'rosenbrock']),
                'max_evals': int(100 + difficulty * 400)
            }
        
        else:
            return {'difficulty': difficulty}
    
    def provide_feedback(self, task_id: str, performance: float):
        """Provide feedback on task performance"""
        # Record performance
        self.task_performances[task_id].append(performance)
        
        # Extract skill from task
        for gen_task in self.generated_tasks:
            if gen_task['task']['task_id'] == task_id:
                skill = gen_task['skill']
                self.update_capability(skill, performance)
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics"""
        return {
            'total_tasks_generated': len(self.generated_tasks),
            'capabilities_tracked': len(self.capabilities),
            'curiosity_distribution': dict(self.curiosity_scores),
            'task_performances': {
                task_id: {
                    'mean': np.mean(perfs),
                    'std': np.std(perfs),
                    'count': len(perfs)
                }
                for task_id, perfs in self.task_performances.items()
                if len(perfs) > 0
            }
        }
    
    def get_top_curious_skills(self, n: int = 5) -> List[tuple]:
        """Get top N most curious skills"""
        if not self.curiosity_scores:
            return []
        
        sorted_skills = sorted(
            self.curiosity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_skills[:n]


if __name__ == "__main__":
    # Test curiosity task generator
    print("🔍 Testing Curiosity Task Generator...")
    
    gen = CuriosityTaskGenerator()
    
    # Simulate agent learning
    current_caps = {
        'visual_recognition': 0.3,
        'sequential_reasoning': 0.5,
        'motor_control': 0.7,
        'pattern_recognition': 0.2
    }
    
    print("\n📊 Current capabilities:")
    for skill, prof in current_caps.items():
        print(f"   {skill}: {prof:.2f}")
    
    print("\n🎯 Generating 10 tasks...")
    for i in range(10):
        task = gen.generate_next_task(current_caps)
        print(f"\n   Task {i+1}: {task['task_id']}")
        print(f"      Skill: {task['skill']}")
        print(f"      Difficulty: {task['difficulty']:.2f}")
        print(f"      Type: {task['type']}")
        
        # Simulate performance feedback
        # Performance improves if task is in ZPD
        current_prof = current_caps.get(task['skill'], 0.5)
        if abs(task['difficulty'] - current_prof) < 0.2:
            # Good ZPD - learning happens
            performance = min(1.0, current_prof + 0.1)
        else:
            # Too hard or too easy - little learning
            performance = current_prof + self.rng.uniform(-0.05, 0.05)
        
        gen.provide_feedback(task['task_id'], performance)
        current_caps[task['skill']] = performance
    
    print("\n📈 Top curious skills:")
    for skill, curiosity in gen.get_top_curious_skills(n=5):
        print(f"   {skill}: {curiosity:.3f}")
    
    print("\n📊 Final statistics:")
    stats = gen.get_statistics()
    print(f"   Total tasks: {stats['total_tasks_generated']}")
    print(f"   Capabilities tracked: {stats['capabilities_tracked']}")