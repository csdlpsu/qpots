"""Reproduce the constrained Branin--Currin workflow illustrated in the paper.

This example requires the ``examples`` extra: ``pip install 'qpots[examples]'``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from qpots import Function, QPOTSConfig, QPOTSRunner


def parse_args() -> argparse.Namespace:
    """Parse tutorial runtime and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("qpots_constrained_tutorial.png"))
    return parser.parse_args()


def evaluate_reference_grid(problem: Function, grid_size: int):
    """Evaluate a dense grid used only to visualize feasibility and Pareto structure."""
    axis = torch.linspace(0.0, 1.0, grid_size, device=problem.device, dtype=problem.dtype)
    x1, x2 = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.column_stack((x1.reshape(-1), x2.reshape(-1)))
    evaluation = problem.evaluate_all(points)
    feasible = evaluation.constraints[:, 0] >= 0
    feasible_objectives = evaluation.objectives[feasible]
    pareto = is_non_dominated(feasible_objectives)
    return axis, points, evaluation, feasible, pareto


def plot_result(problem, config, result, grid_size: int, output: Path) -> None:
    """Plot the observed and reference Pareto sets in output and input space."""
    axis, grid, grid_evaluation, grid_feasible, grid_pareto = evaluate_reference_grid(
        problem, grid_size
    )
    observed_objectives = -result.train_y[:, :2].detach().cpu()
    reference_objectives = -grid_evaluation.objectives[grid_feasible][grid_pareto].cpu()
    reference_inputs = grid[grid_feasible][grid_pareto].cpu()
    constraints = grid_evaluation.constraints[:, 0].reshape(grid_size, grid_size).cpu()
    observed_inputs = result.train_x.detach().cpu()

    initial = slice(0, config.n_initial)
    infill = slice(config.n_initial, None)
    final_batch = result.iterations[-1].candidate_x.detach().cpu()

    figure, (output_axis, input_axis) = plt.subplots(1, 2, figsize=(12, 5))
    output_axis.scatter(*observed_objectives[initial].T, color="#d7a0a0", label="initial")
    output_axis.scatter(*observed_objectives[infill].T, color="#acdce8", label="qPOTS")
    output_axis.scatter(*reference_objectives.T, color="black", marker="+", label="reference")
    output_axis.set(xlabel=r"$f_1$", ylabel=r"$f_2$", title="Output space")
    output_axis.legend()

    input_axis.scatter(*observed_inputs[initial].T, color="#d7a0a0", label="initial")
    input_axis.scatter(*observed_inputs[infill].T, color="#acdce8", label="qPOTS")
    input_axis.scatter(*reference_inputs.T, color="black", marker="+", label="reference")
    input_axis.contour(axis.cpu(), axis.cpu(), constraints, levels=[0.0], colors="#45115f")
    input_axis.scatter(
        *final_batch.T,
        facecolors="none",
        edgecolors="blue",
        marker="s",
        s=120,
        linewidths=2,
        label="final batch",
    )
    input_axis.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Input space", xlim=(0, 1), ylim=(0, 1))
    input_axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main() -> None:
    """Run constrained qPOTS and save the tutorial figure."""
    args = parse_args()
    problem = Function("constrainedbc", dim=2, nobj=2)
    config = QPOTSConfig(
        n_initial=args.n_initial,
        iterations=args.iterations,
        batch_size=args.batch_size,
        n_constraints=1,
        generations=args.generations,
        seed=1023,
    )
    result = QPOTSRunner(problem, config).run()
    plot_result(problem, config, result, args.grid_size, args.output)


if __name__ == "__main__":
    main()
