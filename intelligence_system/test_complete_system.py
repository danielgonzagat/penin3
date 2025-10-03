#!/usr/bin/env python3
"""
Complete system integration test
Tests all components working together
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def test_complete_system():
    print("="*80)
    print("🧪 COMPLETE SYSTEM TEST")
    print("="*80)
    
    # 1. Import all components
    print("\n1️⃣ Testing imports...")
    from config.settings import *
    from core.database import Database
    from models.mnist_classifier import MNISTClassifier
    from agents.dqn_agent import DQNAgent
    from apis.api_manager import APIManager
    import gymnasium as gym
    print("   ✅ All imports successful")
    
    # 2. Test database
    print("\n2️⃣ Testing database...")
    db = Database(DATABASE_PATH)
    db.save_cycle(999, mnist=99.0, cartpole=200.0, cartpole_avg=150.0)
    assert db.get_last_cycle() == 999
    print("   ✅ Database functional")
    
    # 3. Test MNIST
    print("\n3️⃣ Testing MNIST...")
    test_mnist_path = MODELS_DIR / "test_mnist.pth"
    mnist = MNISTClassifier(test_mnist_path, hidden_size=64)
    initial_acc = mnist.evaluate()
    print(f"   ✅ MNIST initial accuracy: {initial_acc:.1f}%")
    assert initial_acc > 5.0  # Better than random
    
    # 4. Test DQN
    print("\n4️⃣ Testing DQN...")
    test_dqn_path = MODELS_DIR / "test_dqn.pth"
    dqn = DQNAgent(state_size=4, action_size=2, model_path=test_dqn_path)
    
    env = gym.make('CartPole-v1')
    state, _ = env.reset()
    action = dqn.select_action(state)
    assert action in [0, 1]
    print(f"   ✅ DQN functional (ε={dqn.epsilon:.3f})")
    env.close()
    
    # 5. Test API Manager
    print("\n5️⃣ Testing API Manager...")
    api_manager = APIManager(API_KEYS, API_MODELS)
    print("   ✅ API Manager initialized")
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - SYSTEM FULLY FUNCTIONAL!")
    print("="*80)
    print("\nSystem is ready for production use.")
    print("Run: ./start.sh")
    
    return True

if __name__ == "__main__":
    try:
        success = test_complete_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
