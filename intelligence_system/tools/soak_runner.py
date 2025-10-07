#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.unified_agi_system import UnifiedAGISystem

def main():
    cycles = int(os.environ.get('SOAK_CYCLES', '200'))
    interval = float(os.environ.get('SOAK_SLEEP', '0'))
    system = UnifiedAGISystem(max_cycles=cycles, use_real_v7=True)
    system.run()
    if interval > 0:
        time.sleep(interval)

if __name__ == '__main__':
    main()
