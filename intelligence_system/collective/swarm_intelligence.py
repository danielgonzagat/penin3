"""
SWARM INTELLIGENCE: Múltiplas instâncias colaborando
Collective intelligence emerges from interaction of many simple agents
"""

import os
import sys
import logging
import threading
import queue
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class SwarmMember:
    """Represents one member of the swarm"""
    member_id: int
    knowledge: Dict[str, Any]
    performance: float
    specialization: str  # What this agent is good at
    active: bool = True


class SharedMemoryBank:
    """
    Shared knowledge base for all swarm members
    Thread-safe distributed memory
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory: List[Dict] = []
        self.lock = threading.Lock()
        self.access_count = 0
        
        logger.info(f"🧠 SharedMemoryBank initialized (capacity={capacity})")
    
    def add(self, agent_id: int, knowledge: Dict[str, Any]):
        """Add knowledge to shared memory"""
        with self.lock:
            self.memory.append({
                'agent_id': agent_id,
                'knowledge': knowledge,
                'timestamp': time.time()
            })
            
            # Keep only most recent
            if len(self.memory) > self.capacity:
                self.memory = self.memory[-self.capacity:]
            
            self.access_count += 1
    
    def get_top_k(self, k: int = 5) -> List[Dict]:
        """Get top K best solutions"""
        with self.lock:
            # Sort by some quality metric (assume 'fitness' or 'performance')
            sorted_memory = sorted(
                self.memory,
                key=lambda x: x['knowledge'].get('fitness', x['knowledge'].get('performance', 0)),
                reverse=True
            )
            
            return sorted_memory[:k]
    
    def get_from_specialist(self, specialization: str) -> Optional[Dict]:
        """Get knowledge from specialist in specific area"""
        with self.lock:
            specialists = [
                m for m in self.memory 
                if m['knowledge'].get('specialization') == specialization
            ]
            
            if specialists:
                # Return best specialist's knowledge
                best = max(specialists, key=lambda x: x['knowledge'].get('performance', 0))
                return best
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        with self.lock:
            return {
                'size': len(self.memory),
                'capacity': self.capacity,
                'access_count': self.access_count,
                'unique_agents': len(set(m['agent_id'] for m in self.memory))
            }


class SwarmIntelligence:
    """
    Swarm of intelligent agents working together
    
    Key principles:
    1. Diversity: Each agent specializes in different areas
    2. Communication: Agents share knowledge via shared memory
    3. Collaboration: Agents help each other learn
    4. Competition: Best solutions propagate
    5. Emergence: Collective intelligence > sum of individuals
    """
    
    def __init__(self, n_agents: int = 10, base_system_class=None):
        """
        Initialize swarm intelligence
        
        Args:
            n_agents: Number of agents in swarm
            base_system_class: Class to instantiate for each agent
        """
        self.n_agents = n_agents
        self.base_system_class = base_system_class
        
        # Swarm members
        self.agents: List[SwarmMember] = []
        
        # Shared infrastructure
        self.shared_memory = SharedMemoryBank(capacity=10000)
        self.task_queue = queue.Queue()
        
        # Statistics
        self.cycles_completed = 0
        self.knowledge_exchanges = 0
        self.emergent_solutions = []
        
        logger.info(f"🐝 SwarmIntelligence initialized: {n_agents} agents")
    
    def initialize_swarm(self):
        """Initialize all swarm members with diverse specializations"""
        logger.info("🌱 Initializing swarm...")
        
        specializations = [
            'visual_recognition',
            'sequential_reasoning',
            'optimization',
            'pattern_recognition',
            'motor_control',
            'memory_tasks',
            'planning',
            'meta_learning',
            'transfer_learning',
            'novelty_seeking'
        ]
        
        for i in range(self.n_agents):
            # Assign specialization
            spec = specializations[i % len(specializations)]
            
            # Create agent
            agent = SwarmMember(
                member_id=i,
                knowledge={},
                performance=0.5,  # Start neutral
                specialization=spec
            )
            
            self.agents.append(agent)
            
            logger.info(f"   🐝 Agent {i}: {spec}")
        
        logger.info(f"   ✅ Swarm initialized: {len(self.agents)} agents")
    
    def distribute_task(self, task: Dict[str, Any]) -> List[Dict]:
        """
        Distribute task to swarm members
        Each agent works on subtask based on specialization
        """
        logger.info(f"📋 Distributing task: {task.get('task_id', 'unknown')}")
        
        results = []
        
        for agent in self.agents:
            if not agent.active:
                continue
            
            # Check if agent's specialization matches task
            task_type = task.get('type', 'general')
            
            if self._matches_specialization(agent.specialization, task_type):
                # Agent works on this task
                result = self._agent_solve(agent, task)
                results.append(result)
                
                # Share knowledge
                self.shared_memory.add(agent.member_id, result)
                self.knowledge_exchanges += 1
        
        # Combine results
        combined = self._combine_results(results)
        
        logger.info(f"   ✅ Task completed by {len(results)} agents")
        
        return results
    
    def _matches_specialization(self, specialization: str, task_type: str) -> bool:
        """Check if agent's specialization matches task"""
        # Simple keyword matching
        return (specialization.lower() in task_type.lower() or 
                task_type.lower() in specialization.lower() or
                task_type == 'general')
    
    def _agent_solve(self, agent: SwarmMember, task: Dict) -> Dict:
        """Agent attempts to solve task"""
        # Simplified: agent uses its current knowledge
        # Real implementation would use actual ML model
        
        # Check shared memory for similar tasks
        similar_solutions = self.shared_memory.get_from_specialist(agent.specialization)
        
        solution_quality = agent.performance
        
        if similar_solutions:
            # Benefit from shared knowledge
            solution_quality += 0.1
        
        result = {
            'agent_id': agent.member_id,
            'task_id': task.get('task_id'),
            'solution_quality': solution_quality,
            'specialization': agent.specialization
        }
        
        return result
    
    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple agents"""
        if not results:
            return {'combined_quality': 0.0}
        
        # Average quality
        avg_quality = sum(r['solution_quality'] for r in results) / len(results)
        
        # Find best individual solution
        best = max(results, key=lambda r: r['solution_quality'])
        
        return {
            'combined_quality': avg_quality,
            'best_agent': best['agent_id'],
            'best_quality': best['solution_quality'],
            'n_contributors': len(results)
        }
    
    def cross_pollinate(self):
        """
        Agents exchange knowledge
        Lower-performing agents learn from higher-performing
        """
        logger.info("🌸 Cross-pollination...")
        
        # Get top solutions
        top_solutions = self.shared_memory.get_top_k(k=5)
        
        if not top_solutions:
            return
        
        # Each agent learns from top solutions
        for agent in self.agents:
            # Learn from best in my specialization
            specialist_knowledge = self.shared_memory.get_from_specialist(agent.specialization)
            
            if specialist_knowledge:
                # Improve performance
                improvement = 0.05 * (specialist_knowledge['knowledge'].get('performance', 0.5) - agent.performance)
                agent.performance = min(1.0, agent.performance + improvement)
            
            # Also learn from general top solutions
            for solution in top_solutions:
                # Small general improvement
                agent.performance = min(1.0, agent.performance + 0.01)
        
        logger.info(f"   ✅ Knowledge exchanged among {len(self.agents)} agents")
    
    def evolve_swarm(self):
        """
        Evolutionary pressure on swarm
        Low performers are replaced or learn from high performers
        """
        logger.info("🧬 Evolving swarm...")
        
        # Sort by performance
        self.agents.sort(key=lambda a: a.performance, reverse=True)
        
        # Top 50% are teachers
        n_teachers = len(self.agents) // 2
        teachers = self.agents[:n_teachers]
        students = self.agents[n_teachers:]
        
        # Students learn from teachers
        for student, teacher in zip(students, teachers):
            # Transfer knowledge
            student.performance = (student.performance + teacher.performance) / 2
            
            # Sometimes change specialization to successful one
            if teacher.performance > student.performance + 0.3:
                if self.rng.random() < 0.2:  # 20% chance
                    logger.info(f"   🔄 Agent {student.member_id} changing spec: {student.specialization} → {teacher.specialization}")
                    student.specialization = teacher.specialization
        
        logger.info(f"   ✅ Swarm evolved: avg_performance={sum(a.performance for a in self.agents) / len(self.agents):.3f}")
    
    def run_swarm_cycle(self, tasks: List[Dict]):
        """
        One full swarm cycle
        
        1. Distribute tasks
        2. Collect results
        3. Cross-pollinate
        4. Evolve
        """
        logger.info(f"\n🐝 SWARM CYCLE {self.cycles_completed}")
        
        # Distribute tasks
        all_results = []
        for task in tasks:
            results = self.distribute_task(task)
            all_results.extend(results)
        
        # Cross-pollination
        self.cross_pollinate()
        
        # Evolution
        if self.cycles_completed % 5 == 0:
            self.evolve_swarm()
        
        self.cycles_completed += 1
        
        return {
            'cycle': self.cycles_completed,
            'tasks_completed': len(tasks),
            'solutions_generated': len(all_results),
            'avg_quality': sum(r['solution_quality'] for r in all_results) / max(len(all_results), 1)
        }
    
    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics"""
        return {
            'n_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents if a.active),
            'cycles_completed': self.cycles_completed,
            'knowledge_exchanges': self.knowledge_exchanges,
            'avg_performance': sum(a.performance for a in self.agents) / max(len(self.agents), 1),
            'specializations': {a.specialization: a.performance for a in self.agents},
            'memory_stats': self.shared_memory.get_statistics()
        }
    
    def detect_emergence(self) -> Optional[Dict]:
        """
        Detect emergent behavior in swarm
        
        Emergent behavior: Swarm solves problems that no individual agent could
        """
        stats = self.get_swarm_statistics()
        avg_perf = stats['avg_performance']
        
        # Check if swarm performance > best individual
        best_individual = max(self.agents, key=lambda a: a.performance)
        
        if avg_perf > best_individual.performance * 1.2:
            # Emergence detected!
            return {
                'type': 'collective_superior',
                'swarm_avg': avg_perf,
                'best_individual': best_individual.performance,
                'emergence_factor': avg_perf / max(best_individual.performance, 0.01)
            }
        
        return None


# Integration helper
def create_swarm_for_v7(n_agents: int = 10):
    """Create swarm intelligence for V7 system"""
    # Note: This would create multiple V7 instances
    # For now, create lightweight agent swarm
    
    swarm = SwarmIntelligence(n_agents=n_agents)
    swarm.initialize_swarm()
    
    return swarm


if __name__ == "__main__":
    # Test swarm intelligence
    print("🐝 Testing Swarm Intelligence...")
    
    swarm = SwarmIntelligence(n_agents=10)
    swarm.initialize_swarm()
    
    # Simulate tasks
    print("\n📋 Running 10 swarm cycles...")
    for cycle in range(10):
        tasks = [
            {'task_id': f'task_{cycle}_0', 'type': 'visual_recognition'},
            {'task_id': f'task_{cycle}_1', 'type': 'optimization'},
            {'task_id': f'task_{cycle}_2', 'type': 'pattern_recognition'}
        ]
        
        result = swarm.run_swarm_cycle(tasks)
        
        if cycle % 3 == 0:
            print(f"\n   Cycle {cycle}: avg_quality={result['avg_quality']:.3f}")
        
        # Check for emergence
        emergence = swarm.detect_emergence()
        if emergence:
            print(f"\n   ✨ EMERGENCE DETECTED: {emergence}")
    
    # Final stats
    print("\n📊 Final swarm statistics:")
    stats = swarm.get_swarm_statistics()
    for key, value in stats.items():
        if key != 'specializations':
            print(f"   {key}: {value}")
    
    print("\n✅ Swarm intelligence test complete")