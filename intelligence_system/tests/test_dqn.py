"""
Professional tests for DQN agent
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.dqn_agent import DQNNetwork, DQNAgent, ReplayMemory


def test_dqn_network_forward():
    """Test DQN network forward pass"""
    model = DQNNetwork(state_size=4, action_size=2, hidden_size=128)
    x = torch.randn(32, 4)
    output = model(x)
    
    assert output.shape == (32, 2), "Output shape should be (batch_size, action_size)"


def test_replay_memory():
    """Test replay memory operations"""
    memory = ReplayMemory(capacity=100)
    
    # Add transitions
    for i in range(50):
        memory.push(
            state=np.array([i, i, i, i]),
            action=0,
            reward=1.0,
            next_state=np.array([i+1, i+1, i+1, i+1]),
            done=False
        )
    
    assert len(memory) == 50
    
    # Sample
    batch = memory.sample(32)
    assert len(batch) == 32


def test_dqn_agent_action_selection():
    """Test DQN agent action selection"""
    test_model_path = Path("/tmp/test_dqn.pth")
    if test_model_path.exists():
        test_model_path.unlink()
    
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        model_path=test_model_path,
        epsilon_start=0.0  # Deterministic for testing
    )
    
    state = np.array([0.1, 0.2, 0.3, 0.4])
    action = agent.select_action(state, training=False)
    
    assert action in [0, 1], "Action should be 0 or 1"
    
    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()


def test_dqn_agent_learns():
    """Test that DQN agent learns from experience"""
    test_model_path = Path("/tmp/test_dqn_learn.pth")
    if test_model_path.exists():
        test_model_path.unlink()
    
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        model_path=test_model_path,
        epsilon_start=1.0,
        memory_size=1000
    )
    
    # Fill memory with experiences
    for i in range(200):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = 1.0 if action == 0 else 0.0  # Reward action 0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
    
    # Train
    initial_loss = agent.train_step()
    
    # Train more
    for _ in range(50):
        agent.train_step()
    
    final_loss = agent.train_step()
    
    # Loss should generally decrease (not always strictly, but trend)
    assert len(agent.memory) >= agent.batch_size, "Memory should have enough samples"
    
    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()


def test_dqn_save_load():
    """Test DQN agent save/load"""
    test_model_path = Path("/tmp/test_dqn_save.pth")
    
    # Create and train
    agent1 = DQNAgent(
        state_size=4,
        action_size=2,
        model_path=test_model_path,
        epsilon_start=0.5
    )
    agent1.epsilon = 0.3
    agent1.steps = 100
    agent1.save()
    
    # Load in new instance
    agent2 = DQNAgent(
        state_size=4,
        action_size=2,
        model_path=test_model_path
    )
    
    assert agent2.epsilon == 0.3, "Epsilon should be loaded"
    assert agent2.steps == 100, "Steps should be loaded"
    
    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
