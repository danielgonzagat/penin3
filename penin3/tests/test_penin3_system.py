import sys
from pathlib import Path
sys.path.insert(0, str(Path('/root/intelligence_system')))
sys.path.insert(0, str(Path('/root/peninaocubo')))

import pytest
from penin3_system import PENIN3System

@pytest.mark.timeout(120)
def test_initial_unified_score_stable():
    p = PENIN3System()
    assert p.state.compute_unified_score() >= 0.99


@pytest.mark.timeout(120)
def test_run_cycle_outputs():
    p = PENIN3System()
    r = p.run_cycle()
    assert 'unified_score' in r
    assert 'v7' in r and 'penin_omega' in r
    assert r['penin_omega']['sigma_valid'] is True


@pytest.mark.timeout(120)
def test_worm_integrity_and_logging():
    p = PENIN3System()
    r = p.run_cycle()
    # Ledger is created and remains valid
    assert p.worm_ledger.verify_integrity() is True


@pytest.mark.timeout(120)
def test_checkpoint_roundtrip(tmp_path):
    p = PENIN3System()
    r1 = p.run_cycle()
    ckpt = tmp_path / 'penin3_state.pkl'
    p.state.save_checkpoint(str(ckpt))

    loaded = PENIN3System.load_checkpoint(str(ckpt))
    assert loaded.state.cycle == p.state.cycle
    assert abs(loaded.state.compute_unified_score() - p.state.compute_unified_score()) < 1e-6
