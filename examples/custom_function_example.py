"""Optimize a constrained cantilever-beam design with a user-defined function.

The analytic beam equations stand in for an external simulation. qPOTS maximizes
model outputs, so the two minimization objectives are negated. The final column
is normalized stress slack, where values greater than or equal to zero are feasible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from qpots import EvaluationResult, Function, QPOTSConfig, QPOTSRunner, RuntimeConfig

BEAM_LENGTH = 1.0  # m
DENSITY = 2700.0  # kg / m^3
YOUNGS_MODULUS = 70.0e9  # Pa
TIP_LOAD = 1000.0  # N
ALLOWABLE_STRESS = 250.0e6  # Pa
REFERENCE_POINT = torch.tensor([-40.0, -0.06], dtype=torch.float64)


def evaluate_cantilever(X: torch.Tensor) -> EvaluationResult:
    """Return negated mass/deflection objectives and nonnegative stress slack."""
    width, height = X.unbind(dim=-1)
    mass = DENSITY * BEAM_LENGTH * width * height
    deflection = 4.0 * TIP_LOAD * BEAM_LENGTH**3 / (YOUNGS_MODULUS * width * height**3)
    bending_stress = 6.0 * TIP_LOAD * BEAM_LENGTH / (width * height**2)
    normalized_slack = (ALLOWABLE_STRESS - bending_stress) / ALLOWABLE_STRESS
    objectives = -torch.stack((mass, deflection), dim=-1)
    return EvaluationResult(objectives=objectives, constraints=normalized_slack.unsqueeze(-1))


def hypervolume_trace(train_y: torch.Tensor, n_initial: int) -> torch.Tensor:
    """Compute feasible hypervolume after each accumulated observation."""
    calculator = Hypervolume(ref_point=REFERENCE_POINT.to(train_y))
    values = []
    for stop in range(n_initial, train_y.shape[0] + 1):
        observed = train_y[:stop]
        feasible = observed[:, 2] >= 0
        if not feasible.any():
            values.append(0.0)
            continue
        objectives = observed[feasible, :2]
        front = objectives[is_non_dominated(objectives)]
        values.append(float(calculator.compute(front)))
    return torch.tensor(values, dtype=torch.float64)


def plot_result(result, config: QPOTSConfig, output: Path) -> None:
    """Plot the feasible objective front and hypervolume history."""
    feasible = result.train_y[:, 2] >= 0
    objectives = -result.train_y[:, :2].detach().cpu()
    feasible_objectives = objectives[feasible.cpu()]
    nondominated = is_non_dominated(-feasible_objectives)
    trace = hypervolume_trace(result.train_y.detach().cpu(), config.n_initial)

    figure, (front_axis, history_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    front_axis.scatter(*objectives[~feasible.cpu()].T, color="#b8b8b8", label="infeasible")
    front_axis.scatter(*feasible_objectives.T, color="#58a6c7", label="feasible")
    front_axis.scatter(
        *feasible_objectives[nondominated].T,
        facecolors="none",
        edgecolors="#93278f",
        s=90,
        label="feasible Pareto set",
    )
    front_axis.set(xlabel="mass [kg]", ylabel="tip deflection [m]", title="Beam designs")
    front_axis.legend()

    evaluations = torch.arange(config.n_initial, result.train_y.shape[0] + 1)
    history_axis.plot(evaluations, trace, color="#45115f", linewidth=2)
    history_axis.axvline(config.n_initial, color="#888888", linestyle="--", linewidth=1)
    history_axis.set(
        xlabel="number of evaluated designs",
        ylabel="feasible hypervolume",
        title="Optimization progress",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    """Parse example runtime and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-initial", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--output", type=Path, default=Path("cantilever_qpots.png"))
    parser.add_argument("--quick", action="store_true", help="Run a small CPU smoke test.")
    return parser.parse_args()


def main() -> None:
    """Run the custom constrained problem and save a progress figure."""
    args = parse_args()
    if args.quick:
        args.n_initial = 8
        args.iterations = 1
        args.batch_size = 1
        args.generations = 2

    runtime = RuntimeConfig(device="cpu", dtype=torch.float64)
    problem = Function(
        name="cantilever",
        dim=2,
        nobj=2,
        combined_func=evaluate_cantilever,
        bounds=torch.tensor([[0.02, 0.04], [0.08, 0.16]], dtype=torch.float64),
        runtime=runtime,
    )
    config = QPOTSConfig(
        n_initial=args.n_initial,
        iterations=args.iterations,
        batch_size=args.batch_size,
        n_constraints=1,
        generations=args.generations,
        seed=args.seed,
    )
    result = QPOTSRunner(problem, config, runtime=runtime).run()
    plot_result(result, config, args.output)


if __name__ == "__main__":
    main()
