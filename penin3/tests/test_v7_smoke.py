import sys
from pathlib import Path
# FIX P1-1: Dynamic path resolution for CI portability
V7_ROOT = Path(__file__).resolve().parents[2] / "intelligence_system"
sys.path.insert(0, str(V7_ROOT))

from core.system_v7_ultimate import IntelligenceSystemV7


def test_v7_smoke_cycle_runs():
    v7 = IntelligenceSystemV7()
    r = v7.run_cycle()
    assert 'mnist' in r and 'cartpole' in r
    assert 'ia3_score' in r
    assert isinstance(r['mnist'].get('test', 0.0), float)
