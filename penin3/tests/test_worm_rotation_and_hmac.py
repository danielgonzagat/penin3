import os
import pytest

from penin.ledger.worm_ledger import WORMLedger


@pytest.mark.timeout(120)
def test_worm_hmac_and_rotation(tmp_path):
    os.environ['PENIN_WORM_HMAC_KEY'] = 'a1' * 16
    test_path = tmp_path / 'worm.jsonl'
    ledger = WORMLedger(str(test_path), rotation_bytes=2000)
    for i in range(40):
        ledger.append('test', f'ev_{i}', {'i': i, 'payload': 'x' * 50})
    valid, err = ledger.verify_chain()
    assert valid, err
    # Attempt rotation (may or may not rotate depending on sizes)
    rotated = ledger.rotate()
    if rotated:
        assert rotated.exists()
    ledger.append('test', 'after_rotate', {'ok': True})
    valid, err = ledger.verify_chain()
    assert valid, err
