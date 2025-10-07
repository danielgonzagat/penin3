"""
OPTION 3: Audit/Test 100+ Cycles
Complete validation of Unified AGI System (V7 + PENIN³ + Synergies)
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.unified_agi_system import UnifiedAGISystem

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print("="*80)
    print(f"🔬 OPTION 3: COMPLETE AUDIT - {cycles} CYCLES")
    print("="*80)
    print("")
    print("System: V7 REAL + PENIN³ + 5 Synergies")
    print("")
    print("What will be tested:")
    print("  ✅ V7 operational evolution (MNIST, CartPole, IA³)")
    print("  ✅ PENIN³ meta evolution (Master I, CAOS, L∞, Sigma)")
    print("  ✅ Synergy amplification (8.50x → 37.5x?)")
    print("  ✅ Consciousness growth")
    print("  ✅ WORM Ledger auditability")
    print("  ✅ Thread communication stability")
    print("")
    print("="*80)
    print("")
    
    # Create system
    print(f"Creating Unified AGI System ({cycles} cycles)...")
    system = UnifiedAGISystem(max_cycles=cycles, use_real_v7=True)
    
    print("")
    print(f"Starting {cycles}-cycle test...")
    print("(Synergies execute every 5 cycles)")
    print("")
    
    start_time = datetime.now()
    
    # Run system
    system.run()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ AUDIT COMPLETE")
    print("="*80)
    
    # Display final state
    final_state = system.unified_state.to_dict()
    print("\nFINAL STATE:")
    print(json.dumps(final_state, indent=2))
    
    # Display synergy summary
    if hasattr(system.penin_orchestrator, 'synergy_orchestrator') and \
       system.penin_orchestrator.synergy_orchestrator:
        synergies = system.penin_orchestrator.synergy_orchestrator
        print(f"\nSYNERGY SUMMARY:")
        print(f"  Total amplification: {synergies.total_amplification:.2f}x")
        print(f"  Executions: {len(synergies.synergy_results)}")
        
        if synergies.synergy_results:
            print("\n  Individual synergies:")
            for r in synergies.synergy_results[:5]:  # Show last 5
                status = "✅" if r.success else "⏳"
                print(f"    {status} {r.synergy_type.value}: {r.amplification:.2f}x")
    
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"Cycles/second: {cycles/duration:.2f}")
    
    print("\n" + "="*80)
