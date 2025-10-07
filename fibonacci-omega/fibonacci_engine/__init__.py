"""
Motor Fibonacci - Universal AI Engine
======================================

A state-of-the-art, plug-and-play optimization and learning engine inspired by
the Fibonacci sequence and golden ratio.

Features:
- Fibonacci scheduling with golden ratio mixing
- Quality-Diversity (MAP-Elites) archive
- Multi-scale spiral search
- UCB bandit meta-control
- Curriculum learning
- WORM ledger for auditability
- Automatic rollback on regression
- Universal adapters for any system

Example:
    >>> from fibonacci_engine import FibonacciEngine, FibonacciConfig
    >>> config = FibonacciConfig()
    >>> engine = FibonacciEngine(config, ...)
    >>> engine.run()
"""

__version__ = "1.0.0"
__author__ = "Fibonacci Engine Team"
__license__ = "MIT"

from fibonacci_engine.core.motor_fibonacci import (
    FibonacciEngine,
    FibonacciConfig,
    Candidate,
)
from fibonacci_engine.core.map_elites import MAPElites
from fibonacci_engine.core.meta_controller import MetaController
from fibonacci_engine.core.curriculum import FibonacciCurriculum
from fibonacci_engine.core.worm_ledger import WormLedger
from fibonacci_engine.core.rollback_guard import RollbackGuard

__all__ = [
    "FibonacciEngine",
    "FibonacciConfig",
    "Candidate",
    "MAPElites",
    "MetaController",
    "FibonacciCurriculum",
    "WormLedger",
    "RollbackGuard",
]
