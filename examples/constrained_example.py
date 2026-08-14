"""Run coupled, constrained qPOTS on the WeldedBeam benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner, RuntimeConfig


def parse_args() -> argparse.Namespace:
    """Parse example runtime and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-initial", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small deterministic CPU configuration for a smoke test.",
    )
    return parser.parse_args()


def report(result):
    print(
        f"Iteration {result.iteration}: candidates={result.candidate_x}; "
        f"observed shape={tuple(result.observed_values.shape)}"
    )


def main() -> None:
    """Run the example and save its observations."""
    args = parse_args()
    if args.quick:
        args.n_initial = 12
        args.iterations = 1
        args.batch_size = 2
        args.generations = 2

    runtime = RuntimeConfig(device="cpu", dtype=torch.float64)
    problem = Function("weldedbeam", dim=4, nobj=2, runtime=runtime)
    config = QPOTSConfig(
        n_initial=args.n_initial,
        iterations=args.iterations,
        batch_size=args.batch_size,
        n_constraints=4,
        generations=args.generations,
        multitask=True,
        seed=args.seed,
    )
    result = QPOTSRunner(problem, config, runtime=runtime, callbacks=[report]).run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.train_x.cpu(), args.output_dir / "weldedbeam_train_x.pt")
    torch.save(result.train_y.cpu(), args.output_dir / "weldedbeam_train_y.pt")
    feasible = (result.train_y[:, 2:] >= 0).all(dim=-1)
    print(f"Feasible observations: {int(feasible.sum())}/{result.train_y.shape[0]}")


if __name__ == "__main__":
    main()
