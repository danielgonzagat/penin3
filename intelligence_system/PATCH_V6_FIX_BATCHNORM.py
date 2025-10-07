"""
PATCH 4: Fix BatchNorm issue in AdvancedPPOAgent
Problem: BatchNorm requires batch_size > 1, but select_action uses single state
Solution: Set network to eval() mode during inference
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("🔧 PATCH 4: Fix BatchNorm in AdvancedPPOAgent")
print("="*80)
print()

# Read current advanced_ppo_agent
with open('agents/advanced_ppo_agent.py', 'r') as f:
    code = f.read()

# Fix 1: Add eval() mode in select_action
old_select = '''    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)'''

new_select = '''    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using policy"""
        # V6 PATCH 4: Set to eval mode for BatchNorm with single sample
        self.network.eval()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)'''

code = code.replace(old_select, new_select)

# Fix 2: Add train() mode in update
old_update_start = '''    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        PPO update with GAE
        """
        if len(self.states_buffer) < self.n_steps:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}'''

new_update_start = '''    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        PPO update with GAE
        V6 PATCH 4: Set to train mode for BatchNorm
        """
        # V6 PATCH 4: Set to train mode for BatchNorm
        self.network.train()
        
        if len(self.states_buffer) < self.n_steps:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}'''

code = code.replace(old_update_start, new_update_start)

# Write patched file
with open('agents/advanced_ppo_agent.py', 'w') as f:
    f.write(code)

print("✅ PATCH 4 APPLIED")
print()
print("Changes:")
print("   1. select_action(): Added self.network.eval() before inference")
print("   2. update(): Added self.network.train() before training")
print()
print("This fixes BatchNorm issue with single-sample inference")
print()
print("="*80)

# Validate
print("🔬 VALIDATING PATCH 4...")
print()

try:
    from agents.advanced_ppo_agent import AdvancedPPOAgent
    from pathlib import Path
    import numpy as np
    
    # Test instantiation
    agent = AdvancedPPOAgent(
        state_size=4,
        action_size=2,
        model_path=Path('models/test.pth'),
        hidden_size=128
    )
    
    # Test select_action with single state
    state = np.array([0.1, 0.2, 0.3, 0.4])
    action, log_prob, value = agent.select_action(state)
    
    print("✅ PATCH 4 VALIDATED")
    print(f"   Test action: {action}")
    print(f"   No BatchNorm error!")
    print()
    
except Exception as e:
    print(f"❌ VALIDATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*80)
print("✅ READY TO RESTART V6")
print("="*80)
