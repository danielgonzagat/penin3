import os, sys, subprocess


def test_worm_cli_inspect(tmp_path):
    path = tmp_path / 'worm.csv'
    head = tmp_path / 'worm_head.txt'
    # Seed ledger
    os.environ['DARWINACCI_WORM_PATH'] = str(path)
    os.environ['DARWINACCI_WORM_HEAD'] = str(head)
    from darwinacci_omega.core.worm import Worm
    w = Worm(path=str(path), head=str(head))
    for i in range(3):
        w.append({'i': i})
    cmd = [sys.executable, '-m', 'darwinacci_omega.scripts.worm_cli', 'inspect', '--path', str(path), '--head', str(head)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert 'HEAD' in res.stdout
