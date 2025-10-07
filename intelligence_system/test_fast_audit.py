"""
FAST AUDIT: 100 cycles with SIMULATED V7 (for speed)
Tests synergies without waiting for real training
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('core.unified_agi_system')
logger.setLevel(logging.INFO)
synergy_logger = logging.getLogger('core.synergies')
synergy_logger.setLevel(logging.INFO)

from core.unified_agi_system import UnifiedAGISystem

if __name__ == "__main__":
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print("="*80)
    print(f"🔬 FAST AUDIT: {cycles} CYCLES (Simulated V7 for speed)")
    print("="*80)
    print("")
    print("Purpose: Test synergies and consciousness evolution")
print("Mode: SIMULATED V7 (fast, ~0.5s/cycle) — use test_100_cycles_real.py for REAL V7")
    print("Synergies: Execute every 5 cycles")
    print("")
    
# Create system with SIMULATED V7 for speed (REAL mode available in test_100_cycles_real.py)
system = UnifiedAGISystem(max_cycles=cycles, use_real_v7=False)
    
    print(f"Running {cycles} cycles...")
    print("")
    
    system.run()
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    state = system.unified_state.to_dict()
    print(f"\nOPERATIONAL (V7 simulated):")
    for k, v in state['operational'].items():
        print(f"  {k}: {v}")
    
    print(f"\nMETA (PENIN³):")
    for k, v in state['meta'].items():
        print(f"  {k}: {v}")
    
    if hasattr(system.penin_orchestrator, 'synergy_orchestrator'):
        syn = system.penin_orchestrator.synergy_orchestrator
        if syn and syn.synergy_results:
            print(f"\nSYNERGIES (last execution):")
            print(f"  Total amplification: {syn.total_amplification:.2f}x")
            for r in syn.synergy_results:
                s = "✅" if r.success else "⏳"
                print(f"  {s} {r.synergy_type.value}: {r.amplification:.2f}x")
    
    print("\n" + "="*80)
    print("✅ FAST AUDIT COMPLETE")
    print("="*80)
