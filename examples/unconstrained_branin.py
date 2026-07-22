"""Run unconstrained qPOTS on the two-objective Branin--Currin benchmark."""

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner

problem = Function("branincurrin", dim=2, nobj=2)
config = QPOTSConfig(
    n_initial=20,
    iterations=100,
    batch_size=1,
    generations=10,
    seed=1023,
)


def report(result):
    print(f"Iteration {result.iteration}: {result.candidate_x}")


result = QPOTSRunner(problem, config, callbacks=[report]).run()
torch.save(result.train_x.cpu(), "train_x.pt")
torch.save(result.train_y.cpu(), "train_y.pt")
