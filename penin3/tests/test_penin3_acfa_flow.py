import sys
from pathlib import Path
# Dynamic path resolution for CI portability
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "intelligence_system"))
sys.path.insert(0, str(ROOT / "peninaocubo"))

from penin3_system import PENIN3System


def test_penin3_acfa_registers_and_evaluates():
    """Test PENIN³ ACFA flow with reduced cycles for CI speed."""
    import os
    p = PENIN3System()
    # Run fewer cycles by default; allow override via env
    max_cycles = int(os.getenv("PENIN3_TEST_CYCLES", "3"))
    for _ in range(max_cycles):
        _ = p.run_cycle()
    status = p.get_status()
    assert status['unified_score'] >= 0.85, f"Unified score {status['unified_score']:.4f} < 0.85"
