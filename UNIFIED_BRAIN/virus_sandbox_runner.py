#!/usr/bin/env python3
"""
Virus Sandbox Runner (Phase 1 - Disabled by default)
- Runs intelligent_virus.py inside a namespaced subprocess with constrained env
- No auto-run on import; explicit CLI required
- Does NOT kill/replace any existing processes
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
import json
from datetime import datetime

ROOT = Path('/root')
LOG = ROOT / 'virus_sandbox.log'


def run_sandbox():
    env = os.environ.copy()
    env['VIRUS_SANDBOX'] = '1'
    env['DISABLE_PROCESS_TERMINATION'] = '1'
    env['SANDBOX_MODE'] = 'soft'

    virus_path = ROOT / 'intelligent_virus.py'
    if not virus_path.exists():
        print(json.dumps({'ok': False, 'error': 'intelligent_virus.py not found'}))
        return 1

    cmd = f"python3 {virus_path} --sandbox"
    with open(LOG, 'a') as lf:
        lf.write(f"\n[{datetime.utcnow().isoformat()}Z] starting sandbox: {cmd}\n")
    proc = subprocess.Popen(cmd, shell=True, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(LOG, 'a') as lf:
        lf.write(f"[{datetime.utcnow().isoformat()}Z] pid={proc.pid}\n")
    print(json.dumps({'ok': True, 'pid': proc.pid, 'log': str(LOG)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(run_sandbox())
