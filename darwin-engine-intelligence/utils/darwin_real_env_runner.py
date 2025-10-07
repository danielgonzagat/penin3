#!/usr/bin/env python3
"""
DARWIN + REAL SELECTIVE ENVIRONMENT RUNNER
- Evolves agents inside RealSelectiveEnvironment
- Logs objective metrics to JSONL
- Updates Darwin checkpoint with best metrics
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from real_selective_environment import RealSelectiveEnvironment
from real_intelligent_agent import RealIntelligentAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RSE-DARWIN - %(levelname)s - %(message)s')
logger = logging.getLogger("RSE-DARWIN")

METRICS_OUT = Path('real_env_metrics.jsonl')
DARWIN_CHECKPOINT = Path('darwin/darwin_checkpoint.json')


def log_metrics(record: Dict[str, Any]):
    with METRICS_OUT.open('a') as f:
        json.dump(record, f)
        f.write('\n')


def update_darwin_checkpoint(best: Dict[str, Any]):
    try:
        ckpt = {}
        if DARWIN_CHECKPOINT.exists():
            ckpt = json.loads(DARWIN_CHECKPOINT.read_text())
        ckpt['real_env_best'] = best
        ckpt['real_env_updated_at'] = datetime.now().isoformat()
        DARWIN_CHECKPOINT.write_text(json.dumps(ckpt, indent=2))
        logger.info("Darwin checkpoint updated with real-env best metrics")
    except Exception as e:
        logger.error(f"Failed to update Darwin checkpoint: {e}")


def summarize_metrics(metrics_path: Path = METRICS_OUT, out_path: Path = Path('real_env_summary.json')) -> Dict[str, Any]:
    """Create summary by 100-cycle blocks with averages and bests."""
    blocks = {}
    last_cycle = 0
    try:
        with metrics_path.open('r') as f:
            for line in f:
                rec = json.loads(line)
                c = int(rec.get('cycle', 0))
                b = (c // 100) * 100
                blk = blocks.setdefault(b, {
                    'cycles': 0,
                    'avg_success_rate': 0.0,
                    'avg_behavior_diversity': 0.0,
                    'max_completed_objectives': 0,
                })
                blk['cycles'] += 1
                blk['avg_success_rate'] += rec.get('communication_success_rate', 0.0)
                blk['avg_behavior_diversity'] += rec.get('behavior_diversity', 0.0)
                blk['max_completed_objectives'] = max(blk['max_completed_objectives'], rec.get('completed_objectives', 0))
                last_cycle = max(last_cycle, c)
    except FileNotFoundError:
        logger.error("Metrics file not found for summarization")
        return {}

    series = []
    for b in sorted(blocks.keys()):
        blk = blocks[b]
        n = max(1, blk['cycles'])
        series.append({
            'block_start_cycle': b,
            'avg_success_rate': blk['avg_success_rate'] / n,
            'avg_behavior_diversity': blk['avg_behavior_diversity'] / n,
            'max_completed_objectives': blk['max_completed_objectives'],
            'samples': n,
        })

    summary = {
        'last_cycle': last_cycle,
        'blocks': series,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {out_path}")
    return summary


def run(num_agents: int = 8, cycles: int = 1000):
    env = RealSelectiveEnvironment(max_agents=num_agents)

    # Seed agents
    for i in range(num_agents):
        agent = RealIntelligentAgent(agent_id=f"agent_{i:03d}")
        env.add_agent(agent)

    best = {
        'best_completed_objectives': 0,
        'best_success_rate': 0.0,
        'best_behavior_diversity': 0.0,
        'at_cycle': 0,
    }

    for c in range(cycles):
        state = env.step()

        # Objective metrics
        objectives = state['objectives']
        comm = state['communication_stats']
        behavior_diversity = len(set(a['action'] for a in state['agent_actions']))
        record = {
            'ts': datetime.now().isoformat(),
            'cycle': state['cycle'],
            'population': state['population'],
            'completed_objectives': objectives['completed_count'],
            'objective_progress': objectives['progress'],
            'communication_success_rate': comm['success_rate'],
            'communication_active_links': comm['active_links'],
            'behavior_diversity': behavior_diversity,
            'emergence_detected': state['emergence_detected'],
        }
        log_metrics(record)

        # Track best
        if objectives['completed_count'] > best['best_completed_objectives']:
            best.update({
                'best_completed_objectives': objectives['completed_count'],
                'best_success_rate': comm['success_rate'],
                'best_behavior_diversity': behavior_diversity,
                'at_cycle': state['cycle'],
            })

        if c % 100 == 0:
            logger.info(f"Cycle {c}: obj_done={objectives['completed_count']} comm_sr={comm['success_rate']:.2f} links={comm['active_links']}")

        if state['emergence_detected']:
            logger.info("Emergence detected — stopping early for analysis")
            break

    update_darwin_checkpoint(best)
    summarize_metrics()
    recompute_best_from_metrics()


def recompute_best_from_metrics(metrics_path: Path = METRICS_OUT) -> Dict[str, Any]:
    """Scan metrics log and recompute overall best, then update checkpoint."""
    best = {
        'best_completed_objectives': 0,
        'best_success_rate': 0.0,
        'best_behavior_diversity': 0.0,
        'at_cycle': 0,
    }
    try:
        with metrics_path.open('r') as f:
            for line in f:
                rec = json.loads(line)
                comp = rec.get('completed_objectives', 0)
                sr = rec.get('communication_success_rate', 0.0)
                beh = rec.get('behavior_diversity', 0.0)
                cyc = rec.get('cycle', 0)

                if (comp > best['best_completed_objectives'] or
                    (comp == best['best_completed_objectives'] and sr > best['best_success_rate']) or
                    (comp == best['best_completed_objectives'] and sr == best['best_success_rate'] and beh > best['best_behavior_diversity'])):
                    best['best_completed_objectives'] = comp
                    best['best_success_rate'] = sr
                    best['best_behavior_diversity'] = beh
                    best['at_cycle'] = cyc
    except FileNotFoundError:
        logger.error("Metrics file not found; cannot recompute best")
        return best

    update_darwin_checkpoint(best)
    logger.info(f"Refreshed real-env best from metrics: {best}")
    return best


if __name__ == '__main__':
    run(num_agents=8, cycles=1000)
