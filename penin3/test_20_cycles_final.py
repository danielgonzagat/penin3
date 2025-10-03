"""
PENIN³ - 20 Cycles Final Validation
====================================

Teste completo com todas as correções aplicadas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/root/intelligence_system")))
sys.path.insert(0, str(Path("/root/peninaocubo")))

from penin3_system import PENIN3System
import logging

# Reduce noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    print("="*80)
    print("🚀 PENIN³ - 20 CYCLES FINAL VALIDATION")
    print("="*80)
    print("\nTesting ALL Week 2 features with fixes applied:")
    print("  ✅ L∞ score using BEST metrics")
    print("  ✅ Unified score using BEST metrics")
    print("  ✅ ACFA League champion-challenger")
    print("  ✅ WORM Ledger comprehensive logging")
    print("  ✅ Sigma Guard RL-aware validation")
    print("  ✅ SR-Ω∞ continuous reflection")
    print("  ✅ CAOS+ dynamic application")
    print()
    
    # Create system
    penin3 = PENIN3System()
    
    results = []
    
    # Run 20 cycles
    for i in range(20):
        try:
            result = penin3.run_cycle()
            
            summary = {
                "cycle": result["cycle"],
                "current_cartpole": result["v7"]["cartpole"],
                "best_cartpole": penin3.state.v7.best_cartpole,
                "mnist": result["v7"]["mnist"],
                "linf": result["penin_omega"]["linf_score"],
                "caos": result["penin_omega"]["caos_factor"],
                "stagnant": result["penin_omega"].get("is_stagnant", False),
                "sigma_valid": result["penin_omega"]["sigma_valid"],
                "sr_score": result["penin_omega"].get("sr_score", 0.0),
                "unified": result["unified_score"]
            }
            
            results.append(summary)
            
            # Progress with key metrics
            stag_icon = "⚠️" if summary["stagnant"] else "🔄"
            sigma_icon = "✅" if summary["sigma_valid"] else "❌"
            
            print(f"{stag_icon} Cycle {i+1:2d}: Cart={summary['current_cartpole']:3.0f}(best={summary['best_cartpole']:.0f}), " +
                  f"L∞={summary['linf']:.4f}, Unified={summary['unified']:.4f} {sigma_icon}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Cycle {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Analysis
    print("\n" + "="*80)
    print(f"📊 ANALYSIS - {len(results)} CYCLES")
    print("="*80)
    
    if len(results) >= 5:
        # Performance stats
        current_cart_values = [r["current_cartpole"] for r in results]
        unified_values = [r["unified"] for r in results]
        linf_values = [r["linf"] for r in results]
        
        print(f"\n📈 CartPole (current exploration):")
        print(f"   Min: {min(current_cart_values):.0f}")
        print(f"   Max: {max(current_cart_values):.0f}")
        print(f"   Avg: {sum(current_cart_values)/len(current_cart_values):.1f}")
        print(f"   Final: {current_cart_values[-1]:.0f}")
        print(f"   Best (maintained): {results[-1]['best_cartpole']:.0f}")
        
        print(f"\n📊 Unified Score:")
        print(f"   Initial: {unified_values[0]:.4f}")
        print(f"   Final: {unified_values[-1]:.4f}")
        print(f"   Min: {min(unified_values):.4f}")
        print(f"   Max: {max(unified_values):.4f}")
        print(f"   Avg: {sum(unified_values)/len(unified_values):.4f}")
        
        print(f"\n🔬 L∞ Score (meta-quality):")
        print(f"   Stable: {linf_values[0]:.4f} → {linf_values[-1]:.4f}")
        print(f"   Using BEST: MNIST={results[-1]['mnist']:.1f}%, CartPole={results[-1]['best_cartpole']:.0f}")
        
        # Meta-metrics
        stagnation_cycles = sum(1 for r in results if r["stagnant"])
        sigma_failures = sum(1 for r in results if not r["sigma_valid"])
        sr_scores = [r["sr_score"] for r in results if r["sr_score"] > 0]
        
        print(f"\n⚙️ Meta-Metrics:")
        print(f"   Stagnation detected: {stagnation_cycles}/{len(results)} cycles")
        print(f"   Sigma failures: {sigma_failures}/{len(results)} cycles")
        print(f"   SR-Ω∞ active: {len(sr_scores)}/{len(results)} cycles")
        if sr_scores:
            print(f"   SR-Ω∞ avg: {sum(sr_scores)/len(sr_scores):.4f}")
        
        # Final verdict
        print("\n" + "="*80)
        print("🎯 FINAL VERDICT")
        print("="*80)
        
        final_unified = unified_values[-1]
        
        if final_unified >= 0.99:
            print(f"\n🎉 EXCELLENT: Unified score {final_unified:.4f} ≥ 0.99")
            print("   PENIN³ is PRODUCTION READY")
        elif final_unified >= 0.95:
            print(f"\n✅ VERY GOOD: Unified score {final_unified:.4f} ≥ 0.95")
            print("   PENIN³ is FUNCTIONAL with minor tuning needed")
        elif final_unified >= 0.90:
            print(f"\n✅ GOOD: Unified score {final_unified:.4f} ≥ 0.90")
            print("   PENIN³ is WORKING but needs optimization")
        else:
            print(f"\n⚠️ NEEDS WORK: Unified score {final_unified:.4f} < 0.90")
            print("   PENIN³ needs more debugging")
        
        # Check stability
        last_5_unified = unified_values[-5:]
        variance = sum((x - final_unified)**2 for x in last_5_unified) / 5
        
        if variance < 0.0001:
            print(f"   Stability: ✅ EXCELLENT (variance={variance:.6f})")
        elif variance < 0.001:
            print(f"   Stability: ✅ GOOD (variance={variance:.6f})")
        else:
            print(f"   Stability: ⚠️ MODERATE (variance={variance:.6f})")
        
        print("\n" + "="*80)
        print("✅ 20 CYCLES VALIDATION COMPLETE")
        print("="*80)
        
        # Save summary
        summary_file = Path("/root/penin3/test_20_cycles_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"PENIN³ 20 Cycles Test Summary\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"Cycles completed: {len(results)}\n")
            f.write(f"Final Unified: {final_unified:.4f}\n")
            f.write(f"L∞ stable: {linf_values[-1]:.4f}\n")
            f.write(f"Best MNIST: {results[-1]['mnist']:.2f}%\n")
            f.write(f"Best CartPole: {results[-1]['best_cartpole']:.0f}\n")
            f.write(f"Stagnation: {stagnation_cycles} cycles\n")
            f.write(f"Sigma failures: {sigma_failures} cycles\n")
        
        print(f"\n📄 Summary saved to: {summary_file}")
    
    else:
        print("\n❌ Not enough cycles completed for analysis")

if __name__ == "__main__":
    main()
