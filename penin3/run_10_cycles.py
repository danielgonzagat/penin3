"""
PENIN³ - Run 10 Real Cycles
============================

Test PENIN³ Week 2 features with 10 real cycles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/root/intelligence_system")))
sys.path.insert(0, str(Path("/root/peninaocubo")))

from penin3_system import PENIN3System
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("🚀 PENIN³ - 10 CYCLES TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Testing Week 2 advanced features:")
    logger.info("  ✅ ACFA League full integration")
    logger.info("  ✅ WORM Ledger continuous logging")
    logger.info("  ✅ Sigma Guard real-time validation")
    logger.info("  ✅ SR-Ω∞ continuous reflection")
    logger.info("  ✅ CAOS+ dynamic application")
    logger.info("")
    
    # Create system
    penin3 = PENIN3System()
    
    results_summary = []
    
    # Run 10 cycles
    for i in range(10):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 RUNNING CYCLE {i+1}/10")
        logger.info(f"{'='*80}\n")
        
        try:
            result = penin3.run_cycle()
            
            summary = {
                "cycle": result["cycle"],
                "mnist": result["v7"]["mnist"],
                "cartpole": result["v7"]["cartpole"],
                "linf": result["penin_omega"].get("linf_score", 0.0),
                "caos": result["penin_omega"].get("caos_factor", 1.0),
                "stagnant": result["penin_omega"].get("is_stagnant", False),
                "sigma_valid": result["penin_omega"].get("sigma_valid", True),
                "unified_score": result["unified_score"]
            }
            
            results_summary.append(summary)
            
            logger.info(f"\n✅ Cycle {i+1} complete:")
            logger.info(f"   MNIST: {summary['mnist']:.2f}%")
            logger.info(f"   CartPole: {summary['cartpole']:.1f}")
            logger.info(f"   L∞: {summary['linf']:.4f}")
            logger.info(f"   Unified: {summary['unified_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Cycle {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL SUMMARY - 10 CYCLES")
    logger.info("="*80)
    
    if results_summary:
        logger.info(f"\nCycles completed: {len(results_summary)}/10\n")
        
        # Calculate averages
        avg_mnist = sum(r["mnist"] for r in results_summary) / len(results_summary)
        avg_cartpole = sum(r["cartpole"] for r in results_summary) / len(results_summary)
        avg_linf = sum(r["linf"] for r in results_summary) / len(results_summary)
        avg_unified = sum(r["unified_score"] for r in results_summary) / len(results_summary)
        
        stagnation_count = sum(1 for r in results_summary if r["stagnant"])
        sigma_failures = sum(1 for r in results_summary if not r["sigma_valid"])
        
        logger.info("AVERAGES:")
        logger.info(f"  MNIST: {avg_mnist:.2f}%")
        logger.info(f"  CartPole: {avg_cartpole:.1f}")
        logger.info(f"  L∞: {avg_linf:.4f}")
        logger.info(f"  Unified Score: {avg_unified:.4f}")
        logger.info("")
        logger.info("META-METRICS:")
        logger.info(f"  Stagnation cycles: {stagnation_count}/10")
        logger.info(f"  Sigma failures: {sigma_failures}/10")
        logger.info(f"  Success rate: {(10-sigma_failures)/10*100:.1f}%")
        
        # Check improvement
        if len(results_summary) > 1:
            first_unified = results_summary[0]["unified_score"]
            last_unified = results_summary[-1]["unified_score"]
            improvement = ((last_unified - first_unified) / first_unified) * 100
            
            logger.info("")
            logger.info("IMPROVEMENT:")
            logger.info(f"  First cycle: {first_unified:.4f}")
            logger.info(f"  Last cycle: {last_unified:.4f}")
            logger.info(f"  Change: {improvement:+.2f}%")
        
        logger.info("\n" + "="*80)
        logger.info("✅ 10 CYCLES TEST COMPLETE")
        logger.info("="*80)
        
        # Final verdict
        if avg_unified > 0.95:
            logger.info("\n🎉 EXCELLENT: Unified score > 95%")
        elif avg_unified > 0.90:
            logger.info("\n✅ GOOD: Unified score > 90%")
        elif avg_unified > 0.85:
            logger.info("\n⚠️ ACCEPTABLE: Unified score > 85%")
        else:
            logger.info("\n❌ NEEDS IMPROVEMENT: Unified score < 85%")
    else:
        logger.info("\n❌ NO CYCLES COMPLETED")
    
if __name__ == "__main__":
    main()
