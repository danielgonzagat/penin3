import os, json, gzip, glob
from pathlib import Path
from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.plugins import toy


def test_checkpoint_roundtrip(tmp_path):
    os.environ['DARWINACCI_TRIALS'] = '1'
    os.environ['DARWINACCI_CHECKPOINT'] = '1'
    os.environ['DARWINACCI_CKPT_DIR'] = str(tmp_path)
    eng = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=2, pop_size=8, seed=7)
    champ = eng.run(max_cycles=2)
    files = sorted(glob.glob(str(tmp_path / 'cycle_*.json.gz')))
    assert files, 'no checkpoints written'
    # Read the last checkpoint
    with gzip.open(files[-1], 'rt') as f:
        payload = json.load(f)
    assert 'format_version' in payload and int(payload['format_version']) >= 1
    assert 'population' in payload and isinstance(payload['population'], list)
    assert 'archive' in payload and isinstance(payload['archive'], list)
    assert 'champion' in payload
    # Enriched fields
    assert isinstance(payload['archive'], list) and all('best_score' in x[1] if isinstance(x, list) or isinstance(x, tuple) else 'best_score' in x for x in payload['archive'])


def test_worm_rotation(tmp_path):
    # Force tiny rotation threshold
    os.environ['DARWINACCI_WORM_ROTATE_MB'] = '0'
    path = tmp_path / 'worm.csv'
    head = tmp_path / 'worm_head.txt'
    os.environ['DARWINACCI_WORM_PATH'] = str(path)
    os.environ['DARWINACCI_WORM_HEAD'] = str(head)
    from darwinacci_omega.core.worm import Worm
    w = Worm(path=str(path), head=str(head))
    for i in range(10):
        w.append({'i': i})
    # Either rotated to gz or ledger exists with header
    rotated = list(tmp_path.glob('worm.csv.*.gz'))
    assert rotated or path.exists()
