"""Optimize a user-defined two-objective function with qPOTS."""

import torch

from qpots import Function, QPOTSConfig, QPOTSRunner


def custom_function(X: torch.Tensor) -> torch.Tensor:
    """Return negated objectives because qPOTS maximizes its model outputs."""
    f1 = X[:, 0] ** 2 + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 1] ** 2
    f2 = (X[:, 0] - 1) ** 2 + (X[:, 1] - 1) ** 2
    return -torch.stack((f1, f2), dim=-1)


problem = Function(
    dim=2,
    nobj=2,
    custom_func=custom_function,
    bounds=torch.tensor([[-5.0, 0.0], [10.0, 15.0]]),
)
config = QPOTSConfig(
    n_initial=20,
    iterations=100,
    batch_size=1,
    generations=10,
    seed=1023,
)
result = QPOTSRunner(problem, config).run()

torch.save(result.train_x.cpu(), "train_x.pt")
torch.save(result.train_y.cpu(), "train_y.pt")
