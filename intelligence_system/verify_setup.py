"""
Verify system setup before starting
Run this to check everything is working
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def verify_setup():
    print("="*80)
    print("🔍 SYSTEM VERIFICATION")
    print("="*80)
    
    errors = []
    
    # 1. Check imports
    print("\n1️⃣ Checking imports...")
    try:
        import config.settings as settings
        print("   ✅ Config")
    except Exception as e:
        errors.append(f"Config import: {e}")
        print(f"   ❌ Config: {e}")
    
    try:
        from core.database import Database
        print("   ✅ Database")
    except Exception as e:
        errors.append(f"Database import: {e}")
        print(f"   ❌ Database: {e}")
    
    try:
        from models.mnist_classifier import MNISTClassifier
        print("   ✅ MNIST")
    except Exception as e:
        errors.append(f"MNIST import: {e}")
        print(f"   ❌ MNIST: {e}")
    
    try:
        from agents.dqn_agent import DQNAgent
        print("   ✅ DQN")
    except Exception as e:
        errors.append(f"DQN import: {e}")
        print(f"   ❌ DQN: {e}")
    
    try:
        from apis.api_manager import APIManager
        print("   ✅ API Manager")
    except Exception as e:
        errors.append(f"API Manager import: {e}")
        print(f"   ❌ API Manager: {e}")
    
    # 2. Check directories
    print("\n2️⃣ Checking directories...")
    from config.settings import DATA_DIR, MODELS_DIR, LOGS_DIR
    
    for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        if d.exists():
            print(f"   ✅ {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"   🆕 Created {d}")
    
    # 3. Check database
    print("\n3️⃣ Testing database...")
    try:
        from config.settings import DATABASE_PATH
        from core.database import Database
        
        db = Database(DATABASE_PATH)
        
        # Test save
        db.save_cycle(0, mnist=10.0, cartpole=20.0, cartpole_avg=20.0)
        
        # Test retrieve
        cycle = db.get_last_cycle()
        best = db.get_best_metrics()
        
        print(f"   ✅ Database functional")
        print(f"   ✅ Last cycle: {cycle}")
        print(f"   ✅ Best metrics: {best}")
        
    except Exception as e:
        errors.append(f"Database test: {e}")
        print(f"   ❌ Database: {e}")
    
    # 4. Check PyTorch
    print("\n4️⃣ Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except Exception as e:
        errors.append(f"PyTorch: {e}")
        print(f"   ❌ PyTorch: {e}")
    
    # 5. Check Gymnasium
    print("\n5️⃣ Checking Gymnasium...")
    try:
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        env.close()
        print(f"   ✅ Gymnasium functional")
    except Exception as e:
        errors.append(f"Gymnasium: {e}")
        print(f"   ❌ Gymnasium: {e}")
    
    # Summary
    print("\n" + "="*80)
    if errors:
        print("❌ VERIFICATION FAILED")
        print("="*80)
        for error in errors:
            print(f"  • {error}")
        return False
    else:
        print("✅ VERIFICATION PASSED - SYSTEM READY!")
        print("="*80)
        print("\nRun: ./start.sh")
        return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
