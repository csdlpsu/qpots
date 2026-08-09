from types import SimpleNamespace

import pytest
import torch

from qpots.utils._constraints import (
    constraint_feasibility,
    penalize_infeasible_objectives,
)
from qpots.utils.acq_utils import hypervolume_from_posterior_mean_gp


@pytest.mark.parametrize(
    ("values", "expected_mask"),
    [
        (
            torch.tensor(
                [
                    [1.0, 2.0, 0.0, 0.5],
                    [3.0, 4.0, -0.1, 1.0],
                    [5.0, 6.0, 1.0, 2.0],
                ]
            ),
            torch.tensor([True, False, True]),
        ),
        (
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 0.0, 0.5],
                        [3.0, 4.0, -0.1, 1.0],
                    ],
                    [
                        [5.0, 6.0, 1.0, 2.0],
                        [7.0, 8.0, 1.0, -0.1],
                    ],
                ]
            ),
            torch.tensor([[True, False], [True, False]]),
        ),
    ],
)
def test_constraint_feasibility_preserves_leading_dimensions(values, expected_mask):
    mask = constraint_feasibility(values, ncons=2)

    assert torch.equal(mask, expected_mask)
    assert mask.shape == values.shape[:-1]


def test_constraint_penalty_only_changes_infeasible_objectives():
    values = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.5],
            [3.0, 4.0, -0.1, 1.0],
            [5.0, 6.0, 1.0, 2.0],
        ]
    )
    original = values.clone()

    penalized = penalize_infeasible_objectives(values, nobj=2, ncons=2)

    assert torch.equal(values, original)
    assert torch.equal(penalized[[0, 2]], original[[0, 2]])
    assert torch.equal(penalized[1, :2], torch.full((2,), -1e12))
    assert torch.equal(penalized[..., 2:], original[..., 2:])


def test_constraint_penalty_broadcasts_over_sample_dimension():
    values = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0, 0.5],
                [3.0, 4.0, -0.1, 1.0],
            ]
        ]
    )

    penalized = penalize_infeasible_objectives(values, nobj=2, ncons=2)

    assert penalized.shape == values.shape
    assert torch.equal(penalized[0, 0], values[0, 0])
    assert torch.equal(penalized[0, 1, :2], torch.full((2,), -1e12))


class PosteriorMeanModel(torch.nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.mean = mean.to(dtype=torch.float64)

    def posterior(self, _X):
        return SimpleNamespace(mean=self.mean)


@pytest.mark.parametrize(
    ("mean", "ref_point", "maximize"),
    [
        (
            torch.tensor(
                [
                    [2.0, 1.0, 0.5],
                    [1.0, 2.0, 0.0],
                    [100.0, 100.0, -0.1],
                ]
            ),
            [0.0, 0.0],
            True,
        ),
        (
            torch.tensor(
                [
                    [1.0, 2.0, 0.5],
                    [2.0, 1.0, 0.0],
                    [-100.0, -100.0, -0.1],
                ]
            ),
            [3.0, 3.0],
            False,
        ),
    ],
)
def test_constrained_posterior_mean_hypervolume_excludes_infeasible_points(
    mean, ref_point, maximize
):
    model = PosteriorMeanModel(mean)

    hv = hypervolume_from_posterior_mean_gp(
        model,
        torch.zeros(mean.shape[0], 1),
        ncons=1,
        ref_point=ref_point,
        maximize=maximize,
    )

    assert hv == pytest.approx(3.0)


def test_constrained_posterior_mean_hypervolume_is_zero_without_feasible_points():
    model = PosteriorMeanModel(
        torch.tensor(
            [
                [2.0, 1.0, -0.5],
                [1.0, 2.0, -0.1],
            ]
        )
    )

    hv = hypervolume_from_posterior_mean_gp(
        model,
        torch.zeros(2, 1),
        ncons=1,
        ref_point=[0.0, 0.0],
        maximize=True,
    )

    assert hv.item() == 0.0
