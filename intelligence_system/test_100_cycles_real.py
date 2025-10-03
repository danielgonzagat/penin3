"""
AUDIT: 100 cycles with REAL V7
Runs unified system with use_real_v7=True and reports metrics and synergy summary.
"""

import sys
import logging
from pathlib import Path
import json
import traceback

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('core.unified_agi_system')
logger.setLevel(logging.INFO)
synergy_logger = logging.getLogger('core.synergies')
synergy_logger.setLevel(logging.INFO)

from core.unified_agi_system import UnifiedAGISystem

if __name__ == "__main__":
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("="*80)
    print(f"🔬 REAL AUDIT: {cycles} CYCLES (V7 REAL)")
    print("="*80)
    print("")
    print("Purpose: Validate unified system with REAL V7 training")
    print("Mode: REAL V7")
    print("Synergies: Execute every 2 cycles")
    print("")

    system = UnifiedAGISystem(max_cycles=cycles, use_real_v7=True)

    print(f"Running {cycles} cycles...")
    print("")

    try:
        system.run()
    except Exception as e:
        print(f"\n❌ Fatal error during run: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)

    state = system.unified_state.to_dict()
    print(f"\nOPERATIONAL (V7 REAL):")
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

    # Save results to JSON for later analysis
    try:
        out_dir = Path(__file__).parent / 'data'
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f'audit_results_{cycles}_cycles.json'
        results = {
            'operational': state['operational'],
            'meta': state['meta'],
            'synergies': [
                {
                    'synergy': r.synergy_type.value,
                    'success': r.success,
                    'amplification': r.amplification,
                    'details': r.details,
                }
                for r in (syn.synergy_results if hasattr(system.penin_orchestrator, 'synergy_orchestrator') and syn and syn.synergy_results else [])
            ],
        }
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved results to {out_file}")
    except Exception as e:
        print(f"\n⚠️ Failed to save JSON results: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("✅ REAL AUDIT COMPLETE")
    print("="*80)
