from unittest.mock import Mock

import pytest
import torch

from qpots.function import Function
from qpots.runner import QPOTSConfig, QPOTSRunner


class FakeModel:
    fit_calls = 0
    multitask_fit_calls = 0

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.models = []

    def fit_gp(self):
        type(self).fit_calls += 1

    def fit_multitask_gp(self):
        type(self).multitask_fit_calls += 1


class FakeAcquisition:
    def __init__(self, *args, **kwargs):
        self.config = args[0]

    def qpots(self, bounds, iteration, **kwargs):
        return torch.full(
            (kwargs["q"], kwargs["dim"]),
            0.25 + 0.1 * iteration,
            dtype=bounds.dtype,
            device=bounds.device,
        )


@pytest.fixture(autouse=True)
def reset_counts():
    FakeModel.fit_calls = 0
    FakeModel.multitask_fit_calls = 0


@pytest.fixture
def function():
    return Function(
        dim=1,
        nobj=2,
        bounds=torch.tensor([[-2.0], [2.0]]),
        custom_func=lambda X: torch.column_stack((X[:, 0], X[:, 0] ** 2)),
    )


def make_runner(function, config, **kwargs):
    return QPOTSRunner(
        function,
        config,
        model_factory=FakeModel,
        acquisition_factory=FakeAcquisition,
        **kwargs,
    )


def test_config_validation():
    with pytest.raises(ValueError, match="iterations"):
        QPOTSConfig(iterations=0)
    with pytest.raises(ValueError, match="partial_evaluations"):
        QPOTSConfig(partial_evaluations=True)
    with pytest.raises(ValueError, match="nystrom_choice"):
        QPOTSConfig(nystrom_choice="invalid")


def test_run_fits_exactly_once_per_iteration(function):
    result = make_runner(function, QPOTSConfig(iterations=3, n_initial=2)).run()
    assert FakeModel.fit_calls == 3
    assert result.train_x.shape == (5, 1)
    assert len(result.iterations) == 3


def test_optional_final_refit(function):
    make_runner(
        function,
        QPOTSConfig(iterations=2, n_initial=2, refit_final_model=True),
    ).run()
    assert FakeModel.fit_calls == 3


def test_step_returns_physical_and_normalized_points(function):
    runner = make_runner(function, QPOTSConfig(iterations=1, n_initial=2))
    result = runner.step()
    assert torch.allclose(result.candidate_x_normalized, torch.tensor([[0.25]], dtype=torch.float64))
    assert torch.allclose(result.candidate_x, torch.tensor([[-1.0]], dtype=torch.float64))


def test_initial_physical_points_are_normalized(function):
    runner = make_runner(function, QPOTSConfig(iterations=1, n_initial=2))
    runner.initialize(torch.tensor([[-2.0], [2.0]]))
    result = runner.run()
    assert torch.allclose(
        result.train_x_normalized[:2],
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
    )


def test_seed_makes_initialization_deterministic(function):
    config = QPOTSConfig(iterations=1, n_initial=3, seed=77)
    first = make_runner(function, config)
    second = make_runner(function, config)
    first.initialize()
    second.initialize()
    assert torch.equal(first._train_x_normalized, second._train_x_normalized)


def test_callback_receives_each_iteration(function):
    callback = Mock()
    make_runner(
        function,
        QPOTSConfig(iterations=2, n_initial=2),
        callbacks=[callback],
    ).run()
    assert callback.call_count == 2


class FakePartialAcquisition(FakeAcquisition):
    def qpots(self, bounds, iteration, **kwargs):
        candidates = super().qpots(bounds, iteration, **kwargs)
        tasks = torch.tensor([[0.0, float("nan")]], device=bounds.device)
        return candidates, tasks


def test_partial_multitask_observations_are_masked():
    function = Function(
        dim=1,
        nobj=1,
        bounds=torch.tensor([[0.0], [1.0]]),
        custom_func=lambda X: X[:, :1],
        cons=lambda X: X[:, :1] - 0.5,
    )
    runner = QPOTSRunner(
        function,
        QPOTSConfig(
            iterations=1,
            n_initial=2,
            n_constraints=1,
            multitask=True,
            partial_evaluations=True,
        ),
        model_factory=FakeModel,
        acquisition_factory=FakePartialAcquisition,
    )
    result = runner.run()
    assert FakeModel.multitask_fit_calls == 1
    assert not torch.isnan(result.train_y[-1, 0])
    assert torch.isnan(result.train_y[-1, 1])


def test_constraint_count_must_match_function():
    function = Function(
        dim=1,
        nobj=1,
        bounds=torch.tensor([[0.0], [1.0]]),
        custom_func=lambda X: X[:, :1],
        cons=lambda X: X[:, :1],
    )
    runner = make_runner(function, QPOTSConfig(iterations=1, n_initial=2))
    with pytest.raises(ValueError, match="n_constraints"):
        runner.initialize()
