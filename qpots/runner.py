"""High-level, typed optimization workflow for qPOTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor

from qpots.acquisition import Acquisition
from qpots.config import RuntimeConfig, resolve_runtime, to_runtime
from qpots.function import EvaluationResult, Function
from qpots.model_object import ModelObject


@dataclass(frozen=True)
class QPOTSConfig:
    """Validated settings for a qPOTS optimization run."""

    iterations: int = 20
    n_initial: int = 10
    batch_size: int = 1
    n_constraints: int = 0
    nystrom: bool = False
    nystrom_choice: str = "pareto"
    generations: int = 10
    multitask: bool = False
    partial_evaluations: bool = False
    correlation_threshold: float | None = None
    seed: int = 1023
    refit_final_model: bool = False

    def __post_init__(self) -> None:
        for field_name in ("iterations", "n_initial", "batch_size", "generations"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        if self.n_constraints < 0:
            raise ValueError("n_constraints cannot be negative")
        if self.nystrom_choice not in {"pareto", "random"}:
            raise ValueError("nystrom_choice must be 'pareto' or 'random'")
        if self.partial_evaluations and not self.multitask:
            raise ValueError("partial_evaluations requires multitask=True")

    def acquisition_options(self, function: Function) -> dict[str, Any]:
        """Return the legacy keyword shape consumed by ``Acquisition.qpots``."""
        return {
            "nystrom": int(self.nystrom),
            "iters": self.iterations,
            "nychoice": self.nystrom_choice,
            "dim": function.dim,
            "ngen": self.generations,
            "q": self.batch_size,
            "mt": int(self.multitask),
            "partial_info": int(self.partial_evaluations),
            "threshold": self.correlation_threshold,
        }


@dataclass(frozen=True)
class IterationResult:
    """Observations collected during one optimization iteration."""

    iteration: int
    candidate_x: Tensor
    candidate_x_normalized: Tensor
    evaluations: EvaluationResult
    observed_values: Tensor
    task_ids: Tensor | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Complete data and iteration history returned by :meth:`QPOTSRunner.run`."""

    train_x: Tensor
    train_x_normalized: Tensor
    train_y: Tensor
    iterations: tuple[IterationResult, ...]
    model: ModelObject


ModelFactory = Callable[..., ModelObject]
AcquisitionFactory = Callable[..., Acquisition]
IterationCallback = Callable[[IterationResult], None]


class QPOTSRunner:
    """Coordinate initialization, model fitting, acquisition, and evaluation."""

    def __init__(
        self,
        function: Function,
        config: QPOTSConfig,
        *,
        runtime: RuntimeConfig | None = None,
        model_factory: ModelFactory = ModelObject,
        acquisition_factory: AcquisitionFactory = Acquisition,
        callbacks: tuple[IterationCallback, ...] | list[IterationCallback] = (),
    ) -> None:
        self.function = function
        self.config = config
        self.runtime = resolve_runtime(runtime or function.runtime)
        self.model_factory = model_factory
        self.acquisition_factory = acquisition_factory
        self.callbacks = tuple(callbacks)
        self.bounds = to_runtime(function.get_bounds(), self.runtime.device, self.runtime.dtype)

        self._train_x_normalized: Tensor | None = None
        self._train_y: Tensor | None = None
        self._initial_count = 0
        self._iteration = 0
        self._history: list[IterationResult] = []
        self._model: ModelObject | None = None

    def _evaluate(self, physical_x: Tensor) -> EvaluationResult:
        result = self.function.evaluate_all(physical_x)
        constraints = result.constraints
        actual_constraints = 0 if constraints is None else constraints.shape[-1]
        if actual_constraints != self.config.n_constraints:
            raise ValueError(
                "Configured n_constraints does not match the function output: "
                f"expected {self.config.n_constraints}, got {actual_constraints}"
            )
        return result

    @staticmethod
    def _stack_evaluation(result: EvaluationResult) -> Tensor:
        if result.constraints is None:
            return result.objectives
        return torch.column_stack((result.objectives, result.constraints))

    def initialize(
        self,
        initial_x: Tensor | None = None,
        initial_y: Tensor | EvaluationResult | None = None,
    ) -> None:
        """Initialize the run from physical-domain points or generated random points."""
        if self._train_x_normalized is not None:
            raise RuntimeError("The runner has already been initialized")
        torch.manual_seed(self.config.seed)
        if initial_x is None:
            normalized_x = torch.rand(
                self.config.n_initial,
                self.function.dim,
                device=self.runtime.device,
                dtype=self.runtime.dtype,
            )
            physical_x = unnormalize(normalized_x, self.bounds)
        else:
            physical_x = to_runtime(initial_x, self.runtime.device, self.runtime.dtype)
            if physical_x.ndim != 2 or physical_x.shape[-1] != self.function.dim:
                raise ValueError("initial_x must have shape (n, function.dim)")
            if torch.any(physical_x < self.bounds[0]) or torch.any(physical_x > self.bounds[1]):
                raise ValueError("initial_x must lie within the function bounds")
            normalized_x = normalize(physical_x, self.bounds)

        if initial_y is None:
            stacked_y = self._stack_evaluation(self._evaluate(physical_x))
        elif isinstance(initial_y, EvaluationResult):
            stacked_y = self._stack_evaluation(initial_y)
        else:
            stacked_y = to_runtime(initial_y, self.runtime.device, self.runtime.dtype)
        stacked_y = to_runtime(stacked_y, self.runtime.device, self.runtime.dtype)
        expected_columns = self.function.nobj + self.config.n_constraints
        if stacked_y.shape != (physical_x.shape[0], expected_columns):
            raise ValueError(
                f"initial_y must have shape ({physical_x.shape[0]}, {expected_columns})"
            )

        self._train_x_normalized = normalized_x
        self._train_y = stacked_y
        self._initial_count = physical_x.shape[0]

    def _build_and_fit_model(self) -> ModelObject:
        model = self.model_factory(
            train_x=self._train_x_normalized,
            train_y=self._train_y,
            bounds=self.bounds,
            nobj=self.function.nobj,
            ncons=self.config.n_constraints,
            ntrain=self._initial_count,
            runtime=self.runtime,
        )
        if self.config.multitask:
            model.fit_multitask_gp()
        else:
            model.fit_gp()
        return model

    def _apply_partial_observations(
        self, full_values: Tensor, task_ids: Tensor | None
    ) -> Tensor:
        if not self.config.partial_evaluations:
            return full_values
        if task_ids is None:
            raise RuntimeError("Partial evaluation mode requires acquisition task IDs")
        observed = torch.full_like(full_values, float("nan"))
        for row, selected_tasks in enumerate(task_ids):
            valid_tasks = selected_tasks[~torch.isnan(selected_tasks)].long()
            observed[row, valid_tasks] = full_values[row, valid_tasks]
        return observed

    def step(self) -> IterationResult:
        """Fit the current data, propose a batch, and evaluate that batch once."""
        if self._train_x_normalized is None:
            self.initialize()
        if self._iteration >= self.config.iterations:
            raise StopIteration("All configured optimization iterations are complete")

        self._model = self._build_and_fit_model()
        acquisition = self.acquisition_factory(
            self.function,
            self._model,
            cons=self.function.get_cons(),
            q=self.config.batch_size,
            runtime=self.runtime,
        )
        proposed = acquisition.qpots(
            self.bounds,
            self._iteration,
            **self.config.acquisition_options(self.function),
        )
        if self.config.partial_evaluations:
            candidate_normalized, task_ids = proposed
        else:
            candidate_normalized, task_ids = proposed, None
        candidate_normalized = to_runtime(
            candidate_normalized.reshape(-1, self.function.dim),
            self.runtime.device,
            self.runtime.dtype,
        )
        candidate_x = unnormalize(candidate_normalized, self.bounds)
        evaluations = self._evaluate(candidate_x)
        full_values = self._stack_evaluation(evaluations)
        observed_values = self._apply_partial_observations(full_values, task_ids)

        self._train_x_normalized = torch.row_stack(
            (self._train_x_normalized, candidate_normalized)
        )
        self._train_y = torch.row_stack((self._train_y, observed_values))
        result = IterationResult(
            iteration=self._iteration,
            candidate_x=candidate_x,
            candidate_x_normalized=candidate_normalized,
            evaluations=evaluations,
            observed_values=observed_values,
            task_ids=task_ids,
        )
        self._history.append(result)
        self._iteration += 1
        for callback in self.callbacks:
            callback(result)
        return result

    def run(
        self,
        initial_x: Tensor | None = None,
        initial_y: Tensor | EvaluationResult | None = None,
    ) -> OptimizationResult:
        """Run all remaining iterations and return the accumulated observations."""
        if self._train_x_normalized is None:
            self.initialize(initial_x, initial_y)
        while self._iteration < self.config.iterations:
            self.step()
        if self.config.refit_final_model:
            self._model = self._build_and_fit_model()
        return OptimizationResult(
            train_x=unnormalize(self._train_x_normalized, self.bounds),
            train_x_normalized=self._train_x_normalized,
            train_y=self._train_y,
            iterations=tuple(self._history),
            model=self._model,
        )
