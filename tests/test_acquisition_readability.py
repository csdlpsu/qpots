from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from qpots.acquisition import Acquisition


def make_acquisition():
    function = Mock(dim=2)
    gps = Mock(
        nobj=2,
        ncons=0,
        bounds=torch.tensor([[0.0, 0.0], [2.0, 4.0]]),
        train_x=torch.zeros(2, 2),
        train_y=torch.zeros(2, 2),
    )
    return Acquisition(function, gps, device="cpu")


def options(**overrides):
    values = {
        "nystrom": 0,
        "mt": 0,
        "partial_info": 0,
        "iters": 10,
        "dim": 2,
        "nychoice": "random",
        "q": 1,
        "ngen": 2,
    }
    values.update(overrides)
    return values


def test_qpots_normalizes_selected_physical_candidates():
    acquisition = make_acquisition()
    pareto_result = SimpleNamespace(X=np.array([[0.5, 1.0], [1.0, 2.0]]))
    acquisition._optimize_qpots_posterior = Mock(return_value=pareto_result)
    selected = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    with patch("qpots.acquisition.select_candidates", return_value=selected):
        result = acquisition.qpots(acquisition.gps.bounds, 3, **options())

    assert torch.equal(result, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    acquisition._optimize_qpots_posterior.assert_called_once()


def test_qpots_partial_mode_preserves_task_ids():
    acquisition = make_acquisition()
    acquisition._optimize_qpots_posterior = Mock(
        return_value=SimpleNamespace(X=np.array([[1.0, 2.0]]))
    )
    selected = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    task_ids = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)

    with patch(
        "qpots.acquisition.select_candidates_total_correlation",
        return_value=(selected, task_ids),
    ):
        result, returned_ids = acquisition.qpots(
            acquisition.gps.bounds, 3, **options(partial_info=1)
        )

    assert torch.equal(result, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    assert torch.allclose(returned_ids, task_ids, equal_nan=True)
