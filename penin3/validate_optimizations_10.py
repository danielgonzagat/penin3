"""
PENIN³ - Optimization Validation (10 cycles)
Measures: cycle durations, unified stability, Sigma pass rate, WORM integrity.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path('/root/intelligence_system')))
sys.path.insert(0, str(Path('/root/peninaocubo')))

from penin3_system import PENIN3System
from penin.ledger.worm_ledger import WORMLedger
from penin3_config import WORM_LEDGER_PATH


def main():
    print('='*80)
    print('🚀 PENIN³ - OPTIMIZATION VALIDATION (10 cycles)')
    print('='*80)

    p = PENIN3System()

    durations = []
    unified_scores = []
    sigma_pass = 0
    cart_rewards = []

    for i in range(10):
        t0 = time.time()
        r = p.run_cycle()
        t1 = time.time()
        dt = t1 - t0
        durations.append(dt)
        unified_scores.append(r['unified_score'])
        sigma_pass += 1 if r['penin_omega'].get('sigma_valid', False) else 0
        cart_rewards.append(r['v7'].get('cartpole', 0.0))
        print(f"Cycle {i+1:2d}: dt={dt:5.2f}s, Cart={cart_rewards[-1]:5.1f}, Unified={r['unified_score']:.4f}, Sigma={'OK' if r['penin_omega'].get('sigma_valid', False) else 'FAIL'}")

    # WORM integrity
    ledger = WORMLedger(str(WORM_LEDGER_PATH) + '.repaired.jsonl') if str(WORM_LEDGER_PATH).endswith('.db') else WORMLedger(str(WORM_LEDGER_PATH))
    valid, err = ledger.verify_chain()

    print('\n' + '='*80)
    print('📊 SUMMARY')
    print('='*80)
    print(f"Durations: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
    print(f"Unified:   min={min(unified_scores):.4f}, max={max(unified_scores):.4f}, avg={sum(unified_scores)/len(unified_scores):.4f}")
    print(f"Sigma:     {sigma_pass}/10 passed")
    print(f"CartPole:  >=500 in {sum(1 for x in cart_rewards if x>=500)}/10 cycles")
    print(f"WORM OK:   {valid} ({'no error' if valid else err})")

if __name__ == '__main__':
    main()
