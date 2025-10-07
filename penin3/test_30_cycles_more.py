"""
PENIN³ - Additional 30 Cycles
==============================

Continuar de onde paramos (20 cycles) para validar 50 cycles totais.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/root/intelligence_system")))
sys.path.insert(0, str(Path("/root/peninaocubo")))

from penin3_system import PENIN3System
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    print("="*80)
    print("🚀 PENIN³ - ADDITIONAL 30 CYCLES (21-50)")
    print("="*80)
    print("\nContinuing from cycle 20...")
    print("Target: 50 cycles total for production validation\n")
    
    # Create system (will start from cycle 20+)
    penin3 = PENIN3System()
    
    initial_cycle = penin3.state.cycle
    target_cycles = 30
    
    results = []
    
    # Run 30 more cycles
    for i in range(target_cycles):
        try:
            result = penin3.run_cycle()
            
            summary = {
                "cycle": result["cycle"],
                "cartpole": result["v7"]["cartpole"],
                "best_cartpole": penin3.state.v7.best_cartpole,
                "mnist": result["v7"]["mnist"],
                "linf": result["penin_omega"]["linf_score"],
                "unified": result["unified_score"],
                "sigma_valid": result["penin_omega"]["sigma_valid"],
                "sigma_passed": len(result["penin_omega"]["sigma_passed_gates"]),
                "stagnant": result["penin_omega"].get("is_stagnant", False)
            }
            
            results.append(summary)
            
            # Progress indicator
            stag = "⚠️" if summary["stagnant"] else "🔄"
            sigma = "✅" if summary["sigma_valid"] else "❌"
            
            print(f"{stag} Cycle {initial_cycle + i + 1:2d}: Cart={summary['cartpole']:3.0f}, " +
                  f"Unified={summary['unified']:.4f}, Sigma={summary['sigma_passed']}/10 {sigma}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Cycle {initial_cycle + i + 1} ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    print("\n" + "="*80)
    print(f"📊 SUMMARY - {len(results)} ADDITIONAL CYCLES")
    print("="*80)
    
    if len(results) >= 10:
        unified_values = [r["unified"] for r in results]
        sigma_passes = sum(1 for r in results if r["sigma_valid"])
        stagnation = sum(1 for r in results if r["stagnant"])
        cartpole_500 = sum(1 for r in results if r["cartpole"] == 500)
        
        print(f"\n📊 Unified Score:")
        print(f"   Stable: {unified_values[0]:.4f} → {unified_values[-1]:.4f}")
        print(f"   Min: {min(unified_values):.4f}")
        print(f"   Max: {max(unified_values):.4f}")
        
        print(f"\n✅ Sigma Guard:")
        print(f"   Passes: {sigma_passes}/{len(results)} ({sigma_passes/len(results)*100:.1f}%)")
        
        print(f"\n📈 Performance:")
        print(f"   CartPole=500: {cartpole_500}/{len(results)} cycles ({cartpole_500/len(results)*100:.1f}%)")
        print(f"   Stagnation: {stagnation}/{len(results)} cycles")
        
        # Calculate variance
        avg_unified = sum(unified_values) / len(unified_values)
        variance = sum((x - avg_unified)**2 for x in unified_values) / len(unified_values)
        
        print(f"\n🎯 Stability:")
        print(f"   Variance: {variance:.8f}")
        if variance < 0.0001:
            print(f"   Rating: ✅ PERFECT")
        
        # Final verdict
        print("\n" + "="*80)
        print("🎯 FINAL VERDICT (50 CYCLES TOTAL)")
        print("="*80)
        
        if unified_values[-1] >= 0.99 and sigma_passes == len(results):
            print("\n🎉 PRODUCTION READY")
            print(f"   ✅ Unified: {unified_values[-1]:.4f} ≥ 0.99")
            print(f"   ✅ Sigma: {sigma_passes}/{len(results)} perfect")
            print(f"   ✅ Stability: variance={variance:.8f}")
        else:
            print(f"\n⚠️ Needs review:")
            if unified_values[-1] < 0.99:
                print(f"   Unified: {unified_values[-1]:.4f} < 0.99")
            if sigma_passes < len(results):
                print(f"   Sigma failures: {len(results)-sigma_passes}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
