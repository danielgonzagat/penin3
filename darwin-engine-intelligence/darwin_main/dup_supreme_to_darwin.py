#!/usr/bin/env python3
"""
Duplicate IA3 Supreme farm neurons into the Darwin Brain by adding the same
number of MicroNeurons (Net2Wider strategy). Idempotent via import marker.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import torch  # noqa: F401  # ensure torch available before importing brain

# Import Darwin brain
from darwin.neurogenesis import Brain  # type: ignore

SUPREME_CKPT = Path("/root/IA3_SUPREME/neural_farm_checkpoint.json")
BRAIN_SNAPSHOT = Path("/root/darwin_data/snapshots/brain_state.pt")
IMPORT_MARKER = Path("/root/darwin_data/imports/ia3_supreme_last_import.json")
IMPORT_MARKER.parent.mkdir(parents=True, exist_ok=True)


async def read_supreme_population() -> int:
    if not SUPREME_CKPT.exists():
        raise FileNotFoundError(f"Missing IA3 Supreme checkpoint: {SUPREME_CKPT}")
    data = json.loads(SUPREME_CKPT.read_text())
    farms = data.get("farms") or {}
    total = 0
    for k in ("input", "hidden", "output"):
        part = farms.get(k) or {}
        total += int(part.get("population") or 0)
    return await total


async def read_last_import_total() -> int:
    if not IMPORT_MARKER.exists():
        return await 0
    try:
        meta = json.loads(IMPORT_MARKER.read_text())
        return await int(meta.get("last_imported_population") or 0)
    except Exception:
        return await 0


async def write_marker(imported_total: int, added: int):
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "last_imported_population": imported_total,
        "last_added": added,
    }
    IMPORT_MARKER.write_text(json.dumps(payload, indent=2))


async def main():
    supreme_total = read_supreme_population()
    last_imported = read_last_import_total()
    delta = max(0, supreme_total - last_imported)

    # Load or create brain
    brain = Brain.load_or_new(BRAIN_SNAPSHOT)
    pre = len(brain)

    # Add neurons using Net2Wider (duplicates functionally from the strongest gate)
    added = 0
    for _ in range(delta):
        brain.add_neuron()  # add_neuron uses Net2Wider if possible
        added += 1

    # Save snapshot
    brain.save(BRAIN_SNAPSHOT)

    post = len(brain)
    summary = {
        "supreme_population": supreme_total,
        "previously_imported": last_imported,
        "newly_added": added,
        "darwin_neurons_before": pre,
        "darwin_neurons_after": post,
        "snapshot": str(BRAIN_SNAPSHOT),
    }
    logger.info(json.dumps(summary, indent=2))

    # Update marker to current imported total
    if added > 0:
        write_marker(supreme_total, added)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.info(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
