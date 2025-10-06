import os, json, gzip, glob
from pathlib import Path
from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.plugins import toy


def test_restore_enriched_checkpoint(tmp_path):
    os.environ['DARWINACCI_TRIALS'] = '1'
    os.environ['DARWINACCI_CHECKPOINT'] = '1'
    os.environ['DARWINACCI_CKPT_DIR'] = str(tmp_path)
    eng = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=2, pop_size=8, seed=7)
    champ = eng.run(max_cycles=2)
    files = sorted(glob.glob(str(tmp_path / 'cycle_*.json.gz')))
    assert files, 'no checkpoints written'

    # Load via engine
    eng2 = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=1, pop_size=8, seed=999)
    cycle = eng2.load_checkpoint_json(files[-1])
    assert cycle >= 1

    # Validate champion metrics and archive genomes present in payload
    with gzip.open(files[-1], 'rt') as f:
        payload = json.load(f)
    champ_payload = payload.get('champion', {})
    # metrics is optional; if present must be dict
    if 'metrics' in champ_payload and champ_payload['metrics'] is not None:
        assert isinstance(champ_payload['metrics'], dict)
    # archive entries should include genome when available
    for entry in payload.get('archive', []):
        if isinstance(entry, dict):
            assert 'best_score' in entry and 'behavior' in entry
            # genome may be empty dict if not snapshotting; presence is acceptable
            assert 'genome' in entry
        else:
            # legacy tuple/list format
            assert len(entry) == 2
            assert 'best_score' in entry[1]
