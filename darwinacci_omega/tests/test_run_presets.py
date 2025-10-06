import os, sys, subprocess


def test_run_fast_preset(tmp_path):
    os.environ['DARWINACCI_CKPT_DIR'] = str(tmp_path)
    os.environ['DARWINACCI_CHECKPOINT'] = '1'
    cmd = [sys.executable, '-m', 'darwinacci_omega.scripts.run', '--preset', 'fast']
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
