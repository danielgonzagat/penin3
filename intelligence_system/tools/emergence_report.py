#!/usr/bin/env python3
"""
Emergence Report CLI
Summarizes intelligence/emergence metrics across DBs and checkpoints.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

INTEL_DB = Path("/root/intelligence_system/data/intelligence.db")
CONN_DB = Path("/root/system_connections.db")
SURP_DB = Path("/root/emergence_surprises.db")
BRAIN_CKPT_V3 = Path("/root/UNIFIED_BRAIN/real_env_checkpoint_v3.json")
BRAIN_CKPT_V2 = Path("/root/UNIFIED_BRAIN/real_env_checkpoint.json")


def safe_query(db_path: Path, query: str, params: tuple = ()):  # -> list[tuple]
    try:
        if not db_path.exists():
            return []
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(query, params)
            return cur.fetchall()
    except Exception:
        return []


def fmt_dt(ts: int | float | None) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(int(ts)).isoformat()
    except Exception:
        return str(ts)


def main():
    print("=== Emergence Report ===")
    # Intelligence DB
    cyc = safe_query(INTEL_DB, "SELECT COUNT(*), MAX(cycle) FROM cycles")
    best = safe_query(INTEL_DB, "SELECT MAX(mnist_accuracy), MAX(cartpole_avg_reward) FROM cycles")
    print("[intelligence.db]")
    if cyc:
        print(f"  rows={cyc[0][0]}, max_cycle={cyc[0][1]}")
    if best:
        print(f"  best_mnist={best[0][0] or 0:.3f}, best_cartpole_avg={best[0][1] or 0:.3f}")

    # System connections
    conns = safe_query(CONN_DB, "SELECT COUNT(*), MAX(timestamp) FROM connections")
    print("[system_connections.db]")
    if conns:
        print(f"  connections={conns[0][0]}, last_ts={conns[0][1]}")

    # Surprises
    surp_cnt = safe_query(SURP_DB, "SELECT COUNT(*), MAX(surprise_score) FROM surprises")
    top_surp = safe_query(SURP_DB, "SELECT system, metric, surprise_score, timestamp FROM surprises ORDER BY surprise_score DESC LIMIT 5")
    print("[emergence_surprises.db]")
    if surp_cnt:
        print(f"  surprises={surp_cnt[0][0]}, max_score={surp_cnt[0][1] or 0:.3f}")
    for row in top_surp or []:
        system, metric, score, ts = row
        print(f"   • {system}.{metric}: {score:.2f} at {ts}")

    # BLOCO 2 - TAREFA 21: Query brain_metrics
    brain_m = safe_query(INTEL_DB, """
        SELECT 
            COUNT(*) as count,
            AVG(coherence) as avg_coh,
            AVG(novelty) as avg_nov,
            AVG(ia3_signal) as avg_ia3,
            MAX(ia3_signal) as max_ia3
        FROM brain_metrics
        WHERE episode > (SELECT COALESCE(MAX(episode), 0) - 100 FROM brain_metrics)
    """)
    if brain_m and brain_m[0] and brain_m[0][0] > 0:
        print("[brain_metrics last 100 eps]")
        row = brain_m[0]
        print(f"  records={row[0]}")
        print(f"  avg_coherence={row[1]:.3f}, avg_novelty={row[2]:.3f}")
        print(f"  avg_ia3={row[3]:.3f}, max_ia3={row[4]:.3f}")
    
    # Brain checkpoint quick glance
    ck = None
    if BRAIN_CKPT_V3.exists():
        ck = json.loads(BRAIN_CKPT_V3.read_text())
        tag = "v3"
    elif BRAIN_CKPT_V2.exists():
        ck = json.loads(BRAIN_CKPT_V2.read_text())
        tag = "v2"
    else:
        tag = None
    print("[brain_checkpoint]")
    if ck:
        stats = ck.get("stats", {})
        print(f"  tag={tag}, ep={ck.get('episode')}, best_reward={ck.get('best_reward')}")
        print(f"  avg100={stats.get('avg_reward_last_100')}, steps={stats.get('total_steps')}, device={stats.get('device')}")
    else:
        print("  no checkpoint found yet")


if __name__ == "__main__":
    main()
