#!/usr/bin/env python3
"""
Run Darwinacci evolution batches from CLI.

Example:
    SOAK=1 python3 intelligence_system/tools/darwinacci_run.py  
"""
import os
import sys
from pathlib import Path

# Ensure package import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.darwinacci_hub import evolve_once, status


def main():
    cycles = int(os.environ.get("DARWINACCI_CYCLES", "5"))
    for _ in range(cycles):
        stats = evolve_once()
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})
    print({"final_status": status()})


if __name__ == "__main__":
    main()
