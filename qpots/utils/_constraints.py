"""Internal helpers for qPOTS constraint handling."""

import torch
from torch import Tensor


def constraint_feasibility(values: Tensor, ncons: int) -> Tensor:
    """Return a mask whose entries are true when every constraint is nonnegative."""
    if ncons <= 0:
        return torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
    return (values[..., -ncons:] >= 0).all(dim=-1)


def penalize_infeasible_objectives(
    values: Tensor,
    nobj: int,
    ncons: int,
    penalty: float = -1e12,
) -> Tensor:
    """Return a copy with infeasible objective values replaced by ``penalty``."""
    penalized = values.clone()
    if ncons <= 0:
        return penalized

    feasible = constraint_feasibility(values, ncons)
    penalty_value = torch.as_tensor(penalty, device=values.device, dtype=values.dtype)
    penalized[..., :nobj] = torch.where(
        feasible.unsqueeze(-1),
        penalized[..., :nobj],
        penalty_value,
    )
    return penalized
