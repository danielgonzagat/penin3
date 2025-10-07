#!/usr/bin/env python3
"""
Repair WORM Ledger chain by rewriting a clean JSONL with consistent hashes.

Usage:
  python3 tools/repair_worm_ledger.py [ledger_path]

- If no path is provided, defaults to intelligence_system/data/unified_worm.jsonl
- Creates a timestamped backup of the original file before repairing
- Writes repaired copy to <ledger>.repaired.jsonl and atomically swaps on success
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import canonical WORMLedger
try:
    from penin.ledger.worm_ledger import WORMLedger
except Exception as e:
    print(f"❌ Could not import WORMLedger: {e}")
    sys.exit(1)

UTC = timezone.utc

def default_ledger_path() -> Path:
    return Path(__file__).resolve().parent.parent / 'data' / 'unified_worm.jsonl'

def backup_file(path: Path) -> Path:
    ts = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    backup = path.with_suffix(path.suffix + f'.backup_{ts}')
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    ledger_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📜 Repairing WORM ledger: {ledger_path}")
    ledger = WORMLedger(str(ledger_path))

    stats = ledger.get_statistics()
    print(f"   Events: {stats['total_events']}, chain_valid={stats['chain_valid']}")
    if stats['chain_valid']:
        print("✅ Chain already valid. Nothing to repair.")
        return 0

    # Export repaired copy
    repaired_path = ledger_path.with_suffix(ledger_path.suffix + '.repaired.jsonl')
    print(f"   → Writing repaired copy: {repaired_path}")
    repaired = ledger.export_repaired_copy(repaired_path)

    # Verify repaired
    repaired_ledger = WORMLedger(str(repaired))
    repaired_stats = repaired_ledger.get_statistics()
    print(f"   Repaired: events={repaired_stats['total_events']}, chain_valid={repaired_stats['chain_valid']}")
    if not repaired_stats['chain_valid']:
        print(f"❌ Repaired copy still invalid: {repaired_stats.get('chain_error')}\n   Keeping originals untouched.")
        return 2

    # Backup original and swap
    if ledger_path.exists():
        backup = backup_file(ledger_path)
        print(f"   🔐 Backup created: {backup}")
    shutil.move(str(repaired), str(ledger_path))
    print("   🔁 Swapped repaired ledger into place.")

    final_stats = WORMLedger(str(ledger_path)).get_statistics()
    print(f"✅ Repair complete. chain_valid={final_stats['chain_valid']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
