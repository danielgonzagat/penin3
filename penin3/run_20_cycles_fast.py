"""
PENIN³ - 20 Cycles Fast Test
=============================

Teste rápido sem MNIST training (só CartPole) para validar Week 2 features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/root/intelligence_system")))
sys.path.insert(0, str(Path("/root/peninaocubo")))

from penin3_config import PENIN3_CONFIG
from penin3_system import PENIN3System
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disable MNIST training for speed
PENIN3_CONFIG["v7"]["enable_mnist"] = False

def main():
    print("="*80)
    print("🚀 PENIN³ - 20 CYCLES FAST TEST")
    print("="*80)
    print("\nConfiguration:")
    print("  - MNIST training: DISABLED (for speed)")
    print("  - CartPole only: ENABLED")
    print("  - Week 2 features: ALL ENABLED")
    print()
    
    # Create system
    penin3 = PENIN3System(config=PENIN3_CONFIG)
    
    results = []
    
    # Run 20 cycles
    for i in range(20):
        try:
            result = penin3.run_cycle()
            
            summary = {
                "cycle": result["cycle"],
                "cartpole": result["v7"]["cartpole"],
                "linf": result["penin_omega"].get("linf_score", 0.0),
                "caos": result["penin_omega"].get("caos_factor", 1.0),
                "stagnant": result["penin_omega"].get("is_stagnant", False),
                "unified": result["unified_score"]
            }
            
            results.append(summary)
            
            # Progress indicator
            status = "🔄" if not summary["stagnant"] else "⚠️"
            print(f"{status} Cycle {i+1:2d}: CartPole={summary['cartpole']:5.0f}, L∞={summary['linf']:.4f}, Unified={summary['unified']:.4f}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Cycle {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    print("\n" + "="*80)
    print(f"📊 SUMMARY - {len(results)} CYCLES COMPLETED")
    print("="*80)
    
    if len(results) >= 2:
        # Calculate stats
        cartpole_values = [r["cartpole"] for r in results]
        unified_values = [r["unified"] for r in results]
        
        initial_cartpole = cartpole_values[0]
        final_cartpole = cartpole_values[-1]
        max_cartpole = max(cartpole_values)
        avg_cartpole = sum(cartpole_values) / len(cartpole_values)
        
        initial_unified = unified_values[0]
        final_unified = unified_values[-1]
        max_unified = max(unified_values)
        
        stagnation_count = sum(1 for r in results if r["stagnant"])
        
        print(f"\n📈 CartPole Performance:")
        print(f"   Initial: {initial_cartpole:.0f}")
        print(f"   Final:   {final_cartpole:.0f}")
        print(f"   Max:     {max_cartpole:.0f}")
        print(f"   Average: {avg_cartpole:.1f}")
        print(f"   Change:  {final_cartpole - initial_cartpole:+.0f}")
        
        print(f"\n📊 Unified Score:")
        print(f"   Initial: {initial_unified:.4f}")
        print(f"   Final:   {final_unified:.4f}")
        print(f"   Max:     {max_unified:.4f}")
        print(f"   Change:  {final_unified - initial_unified:+.4f}")
        
        print(f"\n⚠️ Stagnation: {stagnation_count}/{len(results)} cycles")
        
        # Check if improving
        if final_cartpole > initial_cartpole * 1.5:
            print("\n✅ EXCELLENT: CartPole improved >50%")
        elif final_cartpole > initial_cartpole:
            print("\n✅ GOOD: CartPole improving")
        elif final_cartpole == initial_cartpole:
            print("\n⚠️ STABLE: No improvement yet")
        else:
            print("\n❌ DECLINING: Performance dropped")
        
        # Check convergence
        last_5 = cartpole_values[-5:] if len(cartpole_values) >= 5 else cartpole_values
        variance = sum((x - avg_cartpole)**2 for x in last_5) / len(last_5)
        
        if variance < 100 and avg_cartpole > 400:
            print("🎯 CONVERGED: High performance + low variance")
        elif variance < 100:
            print("⚠️ STABLE: Low variance but not high performance yet")
        else:
            print("🔄 EXPLORING: Still learning")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
