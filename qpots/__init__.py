"""Public package interface for qPOTS."""

from importlib.metadata import PackageNotFoundError, version

from qpots.acquisition import Acquisition
from qpots.benchmark_registry import available_benchmarks, create_benchmark
from qpots.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    RuntimeConfig,
    get_default_runtime,
    set_default_runtime,
)
from qpots.function import EvaluationResult, Function
from qpots.model_object import ModelObject
from qpots.runner import (
    IterationResult,
    OptimizationResult,
    QPOTSConfig,
    QPOTSRunner,
)
from qpots.tsemo_runner import TSEMORunner

try:
    __version__ = version("qpots")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Acquisition",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "EvaluationResult",
    "Function",
    "IterationResult",
    "ModelObject",
    "OptimizationResult",
    "QPOTSConfig",
    "QPOTSRunner",
    "RuntimeConfig",
    "TSEMORunner",
    "available_benchmarks",
    "create_benchmark",
    "get_default_runtime",
    "set_default_runtime",
]
