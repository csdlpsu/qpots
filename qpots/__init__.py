"""
qPOTS package
"""

from qpots.benchmark_registry import available_benchmarks, create_benchmark
from qpots.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    RuntimeConfig,
    get_default_runtime,
    set_default_runtime,
)
from qpots.function import EvaluationResult, Function
from qpots.runner import (
    IterationResult,
    OptimizationResult,
    QPOTSConfig,
    QPOTSRunner,
)

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
    "IterationResult",
    "OptimizationResult",
    "QPOTSConfig",
    "QPOTSRunner",
]
