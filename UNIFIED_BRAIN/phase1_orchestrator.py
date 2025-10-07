#!/usr/bin/env python3
"""
Phase 1 Orchestrator
- Creates safety backups
- Boots Unified Brain with Phase 1 hooks (already wired in daemon)
- Loads IA3 survivors/checkpoints via awaken_all_systems (idempotent)
- Emits integration_report.json with timestamps and statuses
"""
from __future__ import annotations

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/root')
UB = ROOT / 'UNIFIED_BRAIN'
REPORT = ROOT / 'integration_report.json'
LOG_DIR = UB / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd: str, cwd: Path | None = None, timeout: int | None = None) -> dict:
    start = time.time()
    try:
        cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=True,
                            capture_output=True, text=True, timeout=timeout)
        return {
            'cmd': cmd,
            'code': cp.returncode,
            'stdout': cp.stdout[-4000:],
            'stderr': cp.stderr[-4000:],
            'elapsed_sec': round(time.time() - start, 2)
        }
    except subprocess.TimeoutExpired as e:
        return {'cmd': cmd, 'code': -1, 'stdout': e.stdout or '', 'stderr': 'TIMEOUT', 'elapsed_sec': round(time.time() - start, 2)}


def create_backups() -> dict:
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backups = {}
    # Lightweight backups (metadata + configs); avoid multi-GB tar here
    candidates = [
        UB / 'brain_daemon_real_env.py',
        UB / 'integration_hooks.py',
        ROOT / 'THE_NEEDLE.py',
        ROOT / 'ia3_evolution_V3_report_gen600.json',
    ]
    for p in candidates:
        if p.exists():
            dest = p.with_suffix(p.suffix + f'.bak_{ts}')
            dest.write_text(p.read_text())
            backups[str(p)] = str(dest)
    return backups


def awaken_systems() -> dict:
    cmd = 'python3 awaken_all_systems.py'
    return _run(cmd, cwd=UB, timeout=600)


def start_brain_daemon() -> dict:
    # Start in background; logs to brain_daemon_run.log
    cmd = 'nohup python3 brain_daemon_real_env.py > ../brain_daemon_run.log 2>&1 & echo $!'
    result = _run(cmd, cwd=UB, timeout=20)
    return result


def main():
    report = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'phase': 'Phase1',
        'actions': [],
    }

    os.environ.setdefault('ENABLE_GODEL', '1')
    os.environ.setdefault('ENABLE_NEEDLE_META', '1')

    report['actions'].append({'step': 'create_backups', 'result': create_backups()})

    # Load IA3/Survivors via provided script (idempotent)
    if (UB / 'awaken_all_systems.py').exists():
        report['actions'].append({'step': 'awaken_all_systems', 'result': awaken_systems()})
    else:
        report['actions'].append({'step': 'awaken_all_systems', 'skip': True, 'reason': 'script not found'})

    # Start/ensure brain daemon running with hooks
    report['actions'].append({'step': 'start_brain_daemon', 'result': start_brain_daemon()})

    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({'ok': True, 'report_path': str(REPORT)}, indent=2))


if __name__ == '__main__':
    main()
