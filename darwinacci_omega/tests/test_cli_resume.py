import os, sys, json, gzip, glob, subprocess
from pathlib import Path


def test_cli_from_config_and_resume(tmp_path):
    # Prepare config file
    cfg = {
        "pop_size": 10,
        "max_cycles": 2,
        "seed": 99
    }
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg))

    # Prepare checkpoint dir env
    ck_dir = tmp_path / 'ckpts'
    os.environ['DARWINACCI_CKPT_DIR'] = str(ck_dir)
    os.environ['DARWINACCI_CHECKPOINT'] = '1'
    os.environ['DARWINACCI_TRIALS'] = '1'

    # First run: from-config and write checkpoints
    cmd = [sys.executable, '-m', 'darwinacci_omega.scripts.run', '--from-config', str(cfg_path), '--cycles', '2']
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr

    files = sorted(glob.glob(str(ck_dir / 'cycle_*.json.gz')))
    assert files, 'expected checkpoint files'

    # Second run: resume
    cmd2 = [sys.executable, '-m', 'darwinacci_omega.scripts.run', '--resume', '--cycles', '1']
    res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
    assert res2.returncode == 0, res2.stderr
