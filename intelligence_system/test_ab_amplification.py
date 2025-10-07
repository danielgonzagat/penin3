"""
A/B AMPLIFICATION HARNESS

Runs two 100-cycle experiments:
- Baseline: V7 only (no synergies)
- Treatment: Unified system (V7 + PENIN³ + Synergies)

Then compares final metrics and prints REAL amplification.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')

from core.unified_agi_system import UnifiedAGISystem
from core.system_v7_ultimate import IntelligenceSystemV7


def extract_metrics_from_v7(v7: IntelligenceSystemV7):
    status = v7.get_system_status()
    return {
        'mnist_final': float(status.get('best_mnist', 0.0)),
        'cartpole_final': float(status.get('best_cartpole', 0.0)),
        'ia3_final': float(status.get('ia3_score_calculated', 0.0)),
    }


def run_baseline(cycles: int = 100):
    v7 = IntelligenceSystemV7()
    for _ in range(cycles):
        v7.run_cycle()
    return extract_metrics_from_v7(v7)


def run_treatment(cycles: int = 100):
    system = UnifiedAGISystem(max_cycles=cycles, use_real_v7=True)
    system.run()
    state = system.unified_state.to_dict()
    return {
        'mnist_final': float(state['operational'].get('best_mnist', 0.0)),
        'cartpole_final': float(state['operational'].get('best_cartpole', 0.0)),
        'ia3_final': float(state['operational'].get('ia3_score', 0.0)),
    }


def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("="*80)
    print(f"🔬 A/B AMPLIFICATION TEST ({cycles} cycles)")
    print("="*80)

    print("\nBASELINE: V7 only (no synergies)...")
    baseline = run_baseline(cycles)
    print(f"  MNIST: {baseline['mnist_final']:.1f}%")
    print(f"  CartPole: {baseline['cartpole_final']:.0f}")
    print(f"  IA³: {baseline['ia3_final']:.1f}%")

    print("\nTREATMENT: Unified (V7 + PENIN³ + Synergies)...")
    treatment = run_treatment(cycles)
    print(f"  MNIST: {treatment['mnist_final']:.1f}%")
    print(f"  CartPole: {treatment['cartpole_final']:.0f}")
    print(f"  IA³: {treatment['ia3_final']:.1f}%")

    print("\nAMPLIFICATION (REAL):")
    def safe_div(a, b):
        return (a / b) if b and b > 0 else 0.0

    mnist_amp = safe_div(treatment['mnist_final'], baseline['mnist_final'])
    cartpole_amp = safe_div(treatment['cartpole_final'], baseline['cartpole_final'])
    ia3_amp = safe_div(treatment['ia3_final'], baseline['ia3_final'])

    avg_amp = (mnist_amp + cartpole_amp + ia3_amp) / 3 if (mnist_amp or cartpole_amp or ia3_amp) else 0.0

    print(f"  MNIST: {mnist_amp:.2f}x")
    print(f"  CartPole: {cartpole_amp:.2f}x")
    print(f"  IA³: {ia3_amp:.2f}x")
    print(f"  Average: {avg_amp:.2f}x")

    print("\n" + "="*80)
    print("✅ A/B AMPLIFICATION TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
