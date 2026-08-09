"""Reproduce the qPOTS-Decoupled Branin--Currin documentation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from qpots import Function, ModelObject, QPOTSConfig, QPOTSRunner, RuntimeConfig
from qpots.utils.tc_utils import corr_and_total_correlation


def fit_initial_model(result, config: QPOTSConfig, problem: Function) -> ModelObject:
    """Fit the fully observed initial design for the uncertainty comparison."""
    model = ModelObject(
        train_x=result.train_x_normalized[: config.n_initial],
        train_y=result.train_y[: config.n_initial],
        bounds=problem.get_bounds(),
        nobj=2,
        ncons=0,
        ntrain=config.n_initial,
        runtime=problem.runtime,
    )
    model.fit_multitask_gp()
    return model


def total_correlation_grid(model, grid_size: int, runtime: RuntimeConfig):
    """Evaluate posterior total correlation over a regular two-dimensional grid."""
    axis = torch.linspace(0.0, 1.0, grid_size, device=runtime.device, dtype=runtime.dtype)
    x1, x2 = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.column_stack((x1.reshape(-1), x2.reshape(-1)))
    values = []
    with torch.no_grad():
        for point in points:
            covariance = model.posterior(point.unsqueeze(0)).distribution.covariance_matrix
            _, total_correlation = corr_and_total_correlation(covariance)
            value = float("nan") if total_correlation is None else float(total_correlation)
            values.append(max(value, 0.0))
    return axis.cpu(), torch.tensor(values).reshape(grid_size, grid_size)


def selected_task_points(result, task: int) -> torch.Tensor:
    """Collect infill locations where a specific output task was queried."""
    selected = []
    for step in result.iterations:
        if step.task_ids is None:
            continue
        mask = ~torch.isnan(step.task_ids[:, task])
        if mask.any():
            selected.append(step.candidate_x[mask].detach().cpu())
    return torch.row_stack(selected) if selected else torch.empty((0, 2), dtype=torch.float64)


def plot_figure(result, initial_model, grid_size: int, output: Path) -> None:
    """Plot task observations, posterior total correlation, and uncertainty reduction."""
    final_model = result.model.models[0]
    initial_mtgp = initial_model.models[0]
    axis, total_correlation = total_correlation_grid(final_model, grid_size, initial_model.runtime)
    task_zero = selected_task_points(result, 0)
    task_one = selected_task_points(result, 1)

    diagonal = torch.linspace(
        0.0,
        1.0,
        100,
        device=initial_model.device,
        dtype=initial_model.dtype,
    )
    probe = torch.column_stack((diagonal, diagonal))
    with torch.no_grad():
        initial_std = initial_mtgp.posterior(probe).variance.sqrt().cpu()
        final_std = final_model.posterior(probe).variance.sqrt().cpu()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    initial_x = result.train_x[: initial_model.ntrain].detach().cpu()
    axes[0].scatter(*initial_x.T, color="#b8b8b8", s=22, label="initial: both tasks")
    if task_zero.numel():
        axes[0].scatter(*task_zero.T, marker="o", color="#2979b8", label="Branin queried")
    if task_one.numel():
        axes[0].scatter(*task_one.T, marker="^", color="#d35454", label="Currin queried")
    axes[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Task-specific observations")
    axes[0].legend(fontsize=8)

    contour = axes[1].contourf(axis, axis, total_correlation, levels=16, cmap="viridis")
    figure.colorbar(contour, ax=axes[1], label="total correlation [nats]")
    axes[1].set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Final posterior dependence")

    colors = ("#2979b8", "#d35454")
    labels = ("Branin", "Currin")
    for task, (color, label) in enumerate(zip(colors, labels, strict=True)):
        axes[2].plot(
            diagonal.cpu(),
            initial_std[:, task],
            color=color,
            linestyle="--",
            label=f"{label}: initial",
        )
        axes[2].plot(diagonal.cpu(), final_std[:, task], color=color, label=f"{label}: final")
    axes[2].set(
        xlabel=r"diagonal location $x_1=x_2$",
        ylabel="posterior standard deviation",
        title="Uncertainty reduction",
    )
    axes[2].legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    data_output = output.with_suffix(".npz")
    np.savez(
        data_output,
        train_x=result.train_x.detach().cpu().numpy(),
        train_y=result.train_y.detach().cpu().numpy(),
        task_zero=task_zero.numpy(),
        task_one=task_one.numpy(),
        grid_axis=axis.numpy(),
        total_correlation=total_correlation.numpy(),
        probe=probe.detach().cpu().numpy(),
        initial_std=initial_std.numpy(),
        final_std=final_std.numpy(),
    )
    print(f"Saved {output} and {data_output}")


def parse_args() -> argparse.Namespace:
    """Parse reproduction settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/qpots_doe_total_correlation.png"),
    )
    parser.add_argument("--quick", action="store_true", help="Run a small CPU smoke test.")
    return parser.parse_args()


def main() -> None:
    """Run qPOTS-Decoupled and generate its documentation figure."""
    args = parse_args()
    if args.quick:
        args.n_initial = 8
        args.iterations = 1
        args.batch_size = 1
        args.generations = 2
        args.grid_size = 6

    runtime = RuntimeConfig(device="cpu", dtype=torch.float64)
    problem = Function("branincurrin", dim=2, nobj=2, runtime=runtime)
    config = QPOTSConfig(
        n_initial=args.n_initial,
        iterations=args.iterations,
        batch_size=args.batch_size,
        generations=args.generations,
        multitask=True,
        partial_evaluations=True,
        correlation_threshold=args.threshold,
        seed=args.seed,
        refit_final_model=True,
    )
    result = QPOTSRunner(problem, config, runtime=runtime).run()
    initial_model = fit_initial_model(result, config, problem)
    plot_figure(result, initial_model, args.grid_size, args.output)


if __name__ == "__main__":
    main()
