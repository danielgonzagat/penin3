"""
WORM-like append-only metrics log with state hashes.
- Appends JSON lines with: timestamp, task, seed, reward, config hash, model hash (if available)
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

BASE = Path('.penin_omega/metrics')
BASE.mkdir(parents=True, exist_ok=True)
LOG = BASE / 'worm_metrics.jsonl'


def _hash_dict(d: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    except Exception:
        raw = str(d)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def append_record(task: str, seed: int, reward: float, config: Dict[str, Any], model_state: Dict[str, Any] | None = None) -> None:
    rec = {
        'ts': int(datetime.now().timestamp()),
        'task': task,
        'seed': int(seed),
        'reward': float(reward),
        'config_hash': _hash_dict(config),
        'model_hash': _hash_dict(model_state or {}),
    }
    try:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass
