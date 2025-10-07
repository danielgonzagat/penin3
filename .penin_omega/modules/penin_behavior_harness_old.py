#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω Behavior Harness
Executa tarefas reais (coding com teste sintético, função matemática), avalia automaticamente,
registra resultados em DB e emite métricas para o laço de consciência.
"""
from __future__ import annotations
import os, sys, time, json, sqlite3, tempfile, subprocess, textwrap, random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

ROOT = Path('/root/.penin_omega')
BH_DB = ROOT / 'behavior_metrics.db'
LOG = ROOT / 'logs' / 'behavior_harness.log'


async def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{datetime.utcnow().isoformat()}][BH] {msg}\n")


async def _ensure_db() -> None:
    conn = sqlite3.connect(str(BH_DB))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT,
                variant TEXT,
                success INTEGER,
                score REAL,
                duration_ms INTEGER,
                logs TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


async def task_sum_list_variant(n: int) -> Tuple[bool, float, int, str]:
    # Uses penin_skills.sum_list (currently flawed) and evaluates correctness
    from penin_skills import sum_list as skill_sum_list  # type: ignore
    arr = [random.randint(0, 100) for _ in range(n)]
    expected = sum(arr)
    t0 = time.time()
    logs = ''
    try:
        got = skill_sum_list(arr)
        ok = (got == expected)
        logs = f"got={got} expected={expected}"
    except Exception as e:
        ok = False
        logs = str(e)
    dt = int((time.time() - t0) * 1000)
    score = 1.0 if ok else 0.0
    return await ok, score, dt, logs

async def task_factorial_variant(n: int) -> Tuple[bool, float, int, str]:
    from penin_skills import factorial as skill_factorial  # type: ignore
    import math
    expected = math.factorial(n)
    t0 = time.time()
    logs = ''
    try:
        got = skill_factorial(n)
        ok = (got == expected)
        logs = f"got={got} expected={expected}"
    except Exception as e:
        ok = False
        logs = str(e)
    dt = int((time.time() - t0) * 1000)
    score = 1.0 if ok else 0.0
    return await ok, score, dt, logs


async def run_once() -> Dict[str, Any]:
    tasks = [
        ("sum_list", lambda: task_sum_list_variant(100)),
        ("sum_list", lambda: task_sum_list_variant(200)),
        ("factorial", lambda: task_factorial_variant(5)),
        ("factorial", lambda: task_factorial_variant(7)),
    ]
    results = []
    for name, fn in tasks:
        ok, score, dt, logs = fn()
        results.append((name, f"v{dt}", ok, score, dt, logs))
    conn = sqlite3.connect(str(BH_DB))
    try:
        for name, variant, ok, score, dt, logs in results:
            conn.execute(
                "INSERT INTO results (task_name, variant, success, score, duration_ms, logs, created_at) VALUES (?,?,?,?,?,?,?)",
                (name, variant, int(ok), float(score), int(dt), logs, datetime.utcnow().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()
    agg = {
        'total': len(results),
        'success': sum(1 for r in results if r[2]),
        'avg_score': sum(r[3] for r in results) / max(1, len(results)),
        'avg_ms': sum(r[4] for r in results) / max(1, len(results))
    }
    _log(json.dumps({'agg': agg}))
    return await agg


async def main() -> None:
    _ensure_db()
    _log('Behavior Harness started')
    while True:
        try:
            run_once()
        except Exception as e:
            _log(f'error: {e}')
        time.sleep(10)


if __name__ == '__main__':
    main()
