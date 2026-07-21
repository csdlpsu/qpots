"""Objective and constraint evaluation interfaces used by qPOTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from qpots.benchmark_registry import create_benchmark
from qpots.config import RuntimeConfig, resolve_runtime, to_runtime


@dataclass(frozen=True)
class EvaluationResult:
    """Objectives and optional constraints observed at the same design points."""

    objectives: Tensor
    constraints: Tensor | None = None


class Function:
    """A reusable objective-function interface for qPOTS.

    Bounds always have shape ``(2, dim)``: the first row contains lower bounds
    and the second row contains upper bounds. Subclasses implement
    :meth:`_evaluate`; callable-based users may continue to pass ``custom_func``.
    """

    def __init__(
        self,
        name: str | None = None,
        dim: int = 2,
        nobj: int = 2,
        custom_func: Callable[[Tensor], Tensor] | None = None,
        bounds: Tensor | None = None,
        cons: Callable[[Tensor], Tensor] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        runtime: RuntimeConfig | None = None,
        combined_func: Callable[[Tensor], EvaluationResult] | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        if nobj < 1:
            raise ValueError("nobj must be at least 1")
        if custom_func is not None and combined_func is not None:
            raise ValueError("Pass either custom_func or combined_func, not both")

        self.name = name.lower() if name else None
        self.dim = dim
        self.nobj = nobj
        self.runtime = resolve_runtime(runtime, device=device, dtype=dtype)
        self.device = self.runtime.device
        self.dtype = self.runtime.dtype
        self.custom_func = custom_func
        self.combined_func = combined_func
        self.cons = cons
        self.func: object | None = None

        uses_subclass_hook = type(self)._evaluate is not Function._evaluate
        if custom_func is not None or combined_func is not None or uses_subclass_hook:
            if bounds is None:
                raise ValueError("Custom functions must specify bounds.")
            self.bounds = self._validate_bounds(bounds)
        elif self.name is not None:
            self._initialize_function()
        else:
            raise ValueError("Provide a benchmark name, custom function, or _evaluate subclass")

    def _validate_bounds(self, bounds: Tensor) -> Tensor:
        if not torch.is_tensor(bounds):
            raise TypeError("bounds must be a torch.Tensor with shape (2, dim)")
        resolved = to_runtime(bounds, self.device, self.dtype)
        if resolved.ndim != 2 or resolved.shape != (2, self.dim):
            raise ValueError(
                f"bounds must have shape (2, {self.dim}); got {tuple(resolved.shape)}"
            )
        if not torch.isfinite(resolved).all():
            raise ValueError("bounds must contain only finite values")
        if not torch.all(resolved[0] < resolved[1]):
            raise ValueError("every lower bound must be strictly less than its upper bound")
        return resolved

    def _initialize_function(self) -> None:
        self.func = create_benchmark(self.name or "", dim=self.dim, nobj=self.nobj)
        if hasattr(self.func, "to"):
            self.func = self.func.to(device=self.device, dtype=self.dtype)
        self.bounds = self._validate_bounds(self.func.bounds)
        if hasattr(self.func, "evaluate_slack"):
            self.cons = self._evaluate_builtin_constraints

    def _coerce_input(self, X: Tensor) -> Tensor:
        if not torch.is_tensor(X):
            raise TypeError("X must be a torch.Tensor")
        resolved = to_runtime(X, self.device, self.dtype)
        if resolved.ndim == 0 or resolved.shape[-1] != self.dim:
            raise ValueError(f"X must have final dimension {self.dim}")
        return resolved

    def _coerce_output(self, output: Tensor, X: Tensor, columns: int, label: str) -> Tensor:
        if not torch.is_tensor(output):
            raise TypeError(f"{label} output must be a torch.Tensor")
        resolved = to_runtime(output, self.device, self.dtype)
        if columns == 1 and resolved.shape == X.shape[:-1]:
            resolved = resolved.unsqueeze(-1)
        expected_shape = (*X.shape[:-1], columns)
        if resolved.shape != expected_shape:
            raise ValueError(
                f"{label} output must have shape {expected_shape}; got {tuple(resolved.shape)}"
            )
        return resolved

    def _evaluate(self, X: Tensor) -> Tensor:
        """Evaluate objectives; subclasses override this method."""
        if self.custom_func is not None:
            return self.custom_func(X)
        if self.combined_func is not None:
            return self._call_combined(X).objectives
        if self.func is None:
            raise NotImplementedError("Subclasses must implement _evaluate")
        try:
            return self.func(X)
        except ValueError as err:
            if "within the bounds" not in str(err) or not hasattr(self.func, "evaluate_true"):
                raise
            normalized_input = self.bounds[0] + X * (self.bounds[1] - self.bounds[0])
            return self.func(normalized_input)

    def _evaluate_builtin_constraints(self, X: Tensor) -> Tensor:
        X = self._coerce_input(X)
        try:
            return self.func.evaluate_slack(X)
        except ValueError as err:
            if "within the bounds" not in str(err):
                raise
            physical_input = self.bounds[0] + X * (self.bounds[1] - self.bounds[0])
            return self.func.evaluate_slack(physical_input)

    def _call_combined(self, X: Tensor) -> EvaluationResult:
        result = self.combined_func(X)
        if not isinstance(result, EvaluationResult):
            raise TypeError("combined_func must return an EvaluationResult")
        return result

    def evaluate(self, X: Tensor) -> Tensor:
        """Return objective values with shape ``X.shape[:-1] + (nobj,)``."""
        resolved_input = self._coerce_input(X)
        output = self._evaluate(resolved_input)
        return self._coerce_output(output, resolved_input, self.nobj, "objective")

    def evaluate_all(self, X: Tensor) -> EvaluationResult:
        """Evaluate objectives and constraints together when they are available."""
        resolved_input = self._coerce_input(X)
        if self.combined_func is not None:
            raw_result = self._call_combined(resolved_input)
            objectives = self._coerce_output(
                raw_result.objectives, resolved_input, self.nobj, "objective"
            )
            constraints = raw_result.constraints
            if constraints is not None:
                constraints = self._coerce_constraint_output(constraints, resolved_input)
            return EvaluationResult(objectives, constraints)

        objectives = self._coerce_output(
            self._evaluate(resolved_input), resolved_input, self.nobj, "objective"
        )
        constraint_func = self.get_cons()
        constraints = constraint_func(resolved_input) if constraint_func is not None else None
        return EvaluationResult(objectives, constraints)

    def _coerce_constraint_output(self, output: Tensor, X: Tensor) -> Tensor:
        if not torch.is_tensor(output):
            raise TypeError("constraint output must be a torch.Tensor")
        resolved = to_runtime(output, self.device, self.dtype)
        if resolved.shape[:-1] != X.shape[:-1]:
            raise ValueError("constraint output batch shape must match X")
        return resolved

    def get_bounds(self) -> Tensor:
        """Return bounds with shape ``(2, dim)``."""
        return self.bounds

    def get_cons(self) -> Callable[[Tensor], Tensor] | None:
        """Return the separate constraint callable, if one is available."""
        if self.combined_func is not None:
            def combined_constraints(X: Tensor) -> Tensor:
                resolved_input = self._coerce_input(X)
                constraints = self._call_combined(resolved_input).constraints
                if constraints is None:
                    raise ValueError("combined_func did not return constraints")
                return self._coerce_constraint_output(constraints, resolved_input)

            return combined_constraints
        if self.cons is None:
            return None

        def constraints(X: Tensor) -> Tensor:
            resolved_input = self._coerce_input(X)
            return self._coerce_constraint_output(self.cons(resolved_input), resolved_input)

        return constraints
