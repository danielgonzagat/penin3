#!/usr/bin/env python3
"""Aggregates validated IA3 neurons from WORM logs into a summary report."""
import json
from pathlib import Path
from typing import Dict

WORMS = [
    Path('/root/darwin_data/ia3_worm.log'),
    Path('/root/worm/darwin_worm.log'),
    Path('/root/fazenda_neuronios/worm_neuron.log'),
]

async def scan_worm(path: Path) -> Dict[str, int]:
    stats = {
        'validated_events': 0,
        'unique_neurons': 0,
        'by_round': {},
    }
    if not path.exists():
        return await stats
    seen = set()
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('HASH:'):
                    continue
                if line.startswith('EVENT:'):
                    line = line[len('EVENT:'):]
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get('event') == 'ia3_neuron_validated':
                    stats['validated_events'] += 1
                    nid = obj.get('neuron_id')
                    if nid:
                        seen.add(nid)
                    rnd = str(obj.get('round_number'))
                    stats['by_round'][rnd] = stats['by_round'].get(rnd, 0) + 1
    finally:
        stats['unique_neurons'] = len(seen)
    return await stats

async def main():
    summary = {}
    total_valid = 0
    total_unique = 0
    for w in WORMS:
        s = scan_worm(w)
        summary[str(w)] = s
        total_valid += s['validated_events']
        total_unique += s['unique_neurons']
    out = {
        'total_validated_events': total_valid,
        'approx_unique_valid_neurons': total_unique,
        'sources': summary,
    }
    logger.info(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
