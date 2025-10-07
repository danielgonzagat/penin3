"""Universal adapters for connecting Fibonacci Engine to any system."""

from fibonacci_engine.adapters.rl_synthetic import RLSyntheticAdapter
from fibonacci_engine.adapters.supervised_synthetic import SupervisedSyntheticAdapter
from fibonacci_engine.adapters.tool_pipeline import ToolPipelineAdapter

__all__ = [
    "RLSyntheticAdapter",
    "SupervisedSyntheticAdapter",
    "ToolPipelineAdapter",
]
