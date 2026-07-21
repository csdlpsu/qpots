"""Run decoupled qPOTS oracle selection on the constrained OSY benchmark."""

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner


problem = Function("osy", dim=6, nobj=2)
config = QPOTSConfig(
    n_initial=60,
    iterations=50,
    batch_size=2,
    n_constraints=6,
    generations=20,
    multitask=True,
    partial_evaluations=True,
    correlation_threshold=1e-4,
    seed=1023,
)


def report(result):
    queried = (~torch.isnan(result.observed_values)).sum().item()
    print(
        f"Iteration {result.iteration}: queried "
        f"{queried}/{result.observed_values.numel()} scalar oracles"
    )


result = QPOTSRunner(problem, config, callbacks=[report]).run()
torch.save(result.train_x.cpu(), "osy_train_x.pt")
torch.save(result.train_y.cpu(), "osy_partial_train_y.pt")
