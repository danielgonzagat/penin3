"""
Test Unified AGI System with REAL V7
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.unified_agi_system import UnifiedAGISystem

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTING UNIFIED AGI SYSTEM WITH REAL V7")
    print("="*80)
    print("")
    
    # Create system with REAL V7
    print("Creating Unified AGI System (REAL V7 + PENIN³)...")
    system = UnifiedAGISystem(max_cycles=5, use_real_v7=True)
    
    print("")
    print("Running 5 cycles with REAL V7...")
    print("")
    
    # Run system
    system.run()
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    # Display final state
    final_state = system.unified_state.to_dict()
    print("\nFINAL STATE:")
    for category, values in final_state.items():
        print(f"\n{category.upper()}:")
        for k, v in values.items():
            print(f"  {k}: {v}")
