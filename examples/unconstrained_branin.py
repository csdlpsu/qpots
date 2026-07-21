"""Run qPOTS on a custom two-objective Branin problem."""

import torch
from botorch.test_functions.synthetic import Branin

from qpots import Function, QPOTSConfig, QPOTSRunner


def custom_branin(X: torch.Tensor) -> torch.Tensor:
    values = Branin(negate=True)(X)
    return values.unsqueeze(-1).repeat(1, 2)


problem = Function(
    dim=2,
    nobj=2,
    custom_func=custom_branin,
    bounds=torch.tensor([[-5.0, 0.0], [10.0, 15.0]]),
)
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
