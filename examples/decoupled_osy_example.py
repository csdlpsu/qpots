"""Run decoupled qPOTS oracle selection on the constrained OSY benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner, RuntimeConfig


def parse_args() -> argparse.Namespace:
    """Parse example runtime and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-initial", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the reviewer reproduction settings on CPU.",
    )
    return parser.parse_args()


def report(result):
    queried = (~torch.isnan(result.observed_values)).sum().item()
    print(
        f"Iteration {result.iteration}: queried "
        f"{queried}/{result.observed_values.numel()} scalar oracles"
    )


def main() -> None:
    """Run the example and save its partially observed training data."""
    args = parse_args()
    if args.quick:
        args.n_initial = 12
        args.iterations = 1
        args.batch_size = 2
        args.generations = 2

    runtime = RuntimeConfig(device="cpu", dtype=torch.float64)
    problem = Function("osy", dim=6, nobj=2, runtime=runtime)
    config = QPOTSConfig(
        n_initial=args.n_initial,
        iterations=args.iterations,
        batch_size=args.batch_size,
        n_constraints=6,
        generations=args.generations,
        multitask=True,
        partial_evaluations=True,
        correlation_threshold=args.threshold,
        seed=args.seed,
    )
    result = QPOTSRunner(problem, config, runtime=runtime, callbacks=[report]).run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.train_x.cpu(), args.output_dir / "osy_train_x.pt")
    torch.save(result.train_y.cpu(), args.output_dir / "osy_partial_train_y.pt")
    queried = sum(int((~torch.isnan(step.observed_values)).sum()) for step in result.iterations)
    possible = args.iterations * args.batch_size * result.train_y.shape[-1]
    print(f"Selected {queried}/{possible} possible infill oracle evaluations")


if __name__ == "__main__":
    main()
