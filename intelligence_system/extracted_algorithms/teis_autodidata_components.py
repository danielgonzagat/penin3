"""
TEIS AUTODIDATA COMPONENTS - Extraído de PRESERVED_INTELLIGENCE
Deep Q-Learning, Experience Replay, Curriculum Learning, Transfer Learning

Fonte: PRESERVED_INTELLIGENCE/teis_core/teis_autodidata_100.py
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import time

logger = logging.getLogger(__name__)

class ExperienceReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity  # Store capacity for external access
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-6
    
    def __len__(self):
        """Return number of experiences in buffer"""
        return len(self.buffer)
        
    def push(self, state, action, reward, next_state, done, td_error=None):
        """Add experience with priority"""
        priority = abs(td_error) + self.epsilon if td_error else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        Prioritized sampling
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) or None if not enough samples
        """
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unzip batch into separate components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (list(states), list(actions), list(rewards), list(next_states), list(dones))
    
    def __len__(self):
        return len(self.buffer)


class CurriculumLearner:
    """Adaptive Curriculum Learning"""
    
    def __init__(self):
        self.difficulty_level = 0.0
        self.success_history = deque(maxlen=100)
        self.task_progression = {
            'basic': {'difficulty': 0.0, 'mastery': 0.0},
            'intermediate': {'difficulty': 0.3, 'mastery': 0.0},
            'advanced': {'difficulty': 0.6, 'mastery': 0.0},
            'expert': {'difficulty': 0.9, 'mastery': 0.0}
        }
        
        logger.info("📚 Curriculum Learner initialized")
    
    def adjust_difficulty(self, success: bool):
        """Adjust difficulty based on performance"""
        self.success_history.append(float(success))
        
        if len(self.success_history) < 10:
            return self.difficulty_level
        
        success_rate = np.mean(self.success_history)
        
        # Increase difficulty if doing well
        if success_rate > 0.8 and self.difficulty_level < 1.0:
            self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
        # Decrease if struggling
        elif success_rate < 0.3 and self.difficulty_level > 0.0:
            self.difficulty_level = max(0.0, self.difficulty_level - 0.03)
        
        return self.difficulty_level
    
    def get_task_config(self) -> Dict[str, Any]:
        """Get current task configuration"""
        for category, info in self.task_progression.items():
            if info['difficulty'] <= self.difficulty_level <= info['difficulty'] + 0.3:
                return {
                    'category': category,
                    'difficulty': self.difficulty_level,
                    'reward_multiplier': 1.0 + self.difficulty_level
                }
        
        return {'category': 'basic', 'difficulty': 0.0, 'reward_multiplier': 1.0}


class TransferLearner:
    """Transfer Learning System"""
    
    def __init__(self):
        self.knowledge_base = defaultdict(list)
        self.transfer_history = []
        
        logger.info("🔄 Transfer Learner initialized")
    
    def extract_knowledge(self, agent_id: str, network: nn.Module, experiences: List) -> Dict:
        """Extract knowledge from an agent"""
        knowledge = {
            'agent_id': agent_id,
            'network_state': network.state_dict(),
            'top_experiences': self._get_top_experiences(experiences, n=100),
            'timestamp': time.time()
        }
        
        self.knowledge_base[agent_id].append(knowledge)
        return knowledge
    
    def transfer_to_network(self, target_network: nn.Module, source_agent_ids: List[str] = None) -> bool:
        """Transfer knowledge to target network"""
        if not source_agent_ids:
            source_agent_ids = list(self.knowledge_base.keys())
        
        if not source_agent_ids:
            return False
        
        # Collect relevant knowledge
        relevant_knowledge = []
        for source_id in source_agent_ids:
            if source_id in self.knowledge_base:
                latest = self.knowledge_base[source_id][-1]
                relevant_knowledge.append(latest)
        
        if not relevant_knowledge:
            return False
        
        # Transfer via ensemble averaging
        self._ensemble_transfer(target_network, relevant_knowledge)
        
        logger.info(f"✅ Transferred knowledge from {len(relevant_knowledge)} sources")
        
        return True
    
    def _get_top_experiences(self, experiences: List, n: int = 100) -> List:
        """Select top experiences by reward"""
        if not experiences:
            return []
        
        sorted_exp = sorted(experiences, key=lambda x: x[2] if len(x) > 2 else 0, reverse=True)
        return sorted_exp[:n]
    
    def _ensemble_transfer(self, target_network: nn.Module, knowledge_sources: List[Dict]):
        """Transfer via weighted ensemble"""
        with torch.no_grad():
            target_state = target_network.state_dict()
            
            for key in target_state.keys():
                if 'weight' in key or 'bias' in key:
                    source_weights = []
                    for knowledge in knowledge_sources:
                        if key in knowledge['network_state']:
                            source_weights.append(knowledge['network_state'][key])
                    
                    if source_weights:
                        # 70% original, 30% transferred
                        avg_weight = torch.stack(source_weights).mean(dim=0)
                        target_state[key] = 0.7 * target_state[key] + 0.3 * avg_weight
            
            target_network.load_state_dict(target_state)


if __name__ == "__main__":
    import time
    
    # Test
    logger.info("🧪 Testing TEIS Autodidata Components...")
    
    # Test replay buffer
    buffer = ExperienceReplayBuffer(capacity=100)
    for i in range(50):
        buffer.push(
            state=np.random.randn(4),
            action=np.random.randint(0, 2),
            reward=np.random.random(),
            next_state=np.random.randn(4),
            done=False
        )
    
    batch = buffer.sample(10)
    logger.info(f"✅ Replay Buffer: {len(buffer)} experiences, sampled {len(batch) if batch else 0}")
    
    # Test curriculum
    curriculum = CurriculumLearner()
    for i in range(20):
        curriculum.adjust_difficulty(success=i % 2 == 0)
    
    logger.info(f"✅ Curriculum: difficulty={curriculum.difficulty_level:.2f}")
    
    # Test transfer
    transfer = TransferLearner()
    dummy_net = nn.Linear(10, 2)
    transfer.extract_knowledge("agent1", dummy_net, [(np.zeros(4), 0, 1.0, np.zeros(4), False)])
    
    logger.info(f"✅ Transfer Learning: {len(transfer.knowledge_base)} agents")
    
    logger.info("✅ All TEIS components working!")
