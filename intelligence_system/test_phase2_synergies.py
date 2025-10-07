"""
Test PHASE 2: Core Synergies (V7 + PENIN³)
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.unified_agi_system import UnifiedAGISystem

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTING PHASE 2: CORE SYNERGIES")
    print("="*80)
    print("")
    print("Testing 5 synergies:")
    print("1. Meta-Reasoning + Auto-Coding (2.5x)")
    print("2. Consciousness + Incompletude (2.0x)")
    print("3. Omega Point + Darwin (3.0x)")
    print("4. Self-Reference + Experience Replay (2.0x)")
    print("5. Recursive + MAML (2.5x)")
    print("")
    print("Expected total: up to 37.5x amplification")
    print("")
    print("="*80)
    print("")
    
    # Create system with REAL V7 + PENIN³ + Synergies
    print("Creating Unified AGI System (V7 REAL + PENIN³ + Synergies)...")
    system = UnifiedAGISystem(max_cycles=20, use_real_v7=True)
    
    print("")
    print("Running 20 cycles (synergies execute every 5 cycles)...")
    print("")
    
    # Run system
    system.run()
    
    print("\n" + "="*80)
    print("✅ PHASE 2 TEST COMPLETE")
    print("="*80)
    
    # Display final state
    final_state = system.unified_state.to_dict()
    print("\nFINAL STATE:")
    for category, values in final_state.items():
        print(f"\n{category.upper()}:")
        for k, v in values.items():
            print(f"  {k}: {v}")
    
    # Display synergy summary
    if hasattr(system.penin_orchestrator, 'synergy_orchestrator') and system.penin_orchestrator.synergy_orchestrator:
        synergies = system.penin_orchestrator.synergy_orchestrator
        print(f"\nSYNERGIES:")
        print(f"  Total amplification: {synergies.total_amplification:.2f}x")
        print(f"  Results: {len(synergies.synergy_results)} synergies executed")
