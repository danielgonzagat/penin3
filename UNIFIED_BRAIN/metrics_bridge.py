"""
Metrics bridge: persist external, verifiable metrics to EmergenceTracker.
- Uses sqlite DBs under .penin_omega/metrics/ (created if needed)
- Minimizes coupling: provide simple record_* functions
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading

# Lazy import to avoid heavy deps at import time
_tracker_lock = threading.Lock()
_tracker = None  # type: ignore


def _ensure_dirs() -> Path:
    base = Path(".penin_omega/metrics")
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_tracker():
    """Get a singleton EmergenceTracker instance backed by sqlite files."""
    global _tracker
    if _tracker is not None:
        return _tracker
    with _tracker_lock:
        if _tracker is not None:
            return _tracker
        base = _ensure_dirs()
        from intelligence_system.core.emergence_tracker import EmergenceTracker
        surprises_db = base / "surprises.db"
        connections_db = base / "connections.db"
        _tracker = EmergenceTracker(surprises_db, connections_db)
        return _tracker


def record_reward(value: float, episode: Optional[int] = None) -> None:
    try:
        tr = get_tracker()
        tr.track_metric("reward", float(value), episode=episode)
    except Exception:
        pass


def record_metric(name: str, value: float, episode: Optional[int] = None) -> None:
    try:
        tr = get_tracker()
        tr.track_metric(name, float(value), episode=episode)
    except Exception:
        pass


def record_connection(source: str, target: str, ctype: str = "data_flow", strength: float = 1.0) -> None:
    try:
        tr = get_tracker()
        tr.record_connection(source, target, connection_type=ctype, strength=strength)
    except Exception:
        pass
