"""Proxy module to central Darwin core (single source of truth)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/root/intelligence_system')))
from extracted_algorithms.darwin_engine_real import (
    RealNeuralNetwork,
    Individual,
    DarwinEngine,
    ReproductionEngine,
    DarwinOrchestrator,
)

__all__ = [
    'RealNeuralNetwork',
    'Individual',
    'DarwinEngine',
    'ReproductionEngine',
    'DarwinOrchestrator',
]
