"""
qPOTS package
"""

from qpots.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    RuntimeConfig,
    get_default_runtime,
    set_default_runtime,
)
from qpots.benchmark_registry import available_benchmarks, create_benchmark
from qpots.function import EvaluationResult, Function

__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "RuntimeConfig",
    "get_default_runtime",
    "set_default_runtime",
    "EvaluationResult",
    "Function",
    "available_benchmarks",
    "create_benchmark",
]
