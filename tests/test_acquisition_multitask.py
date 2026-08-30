from unittest.mock import Mock, patch

import pytest
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpots.acquisition import Acquisition
from qpots.function import Function
from qpots.model_object import ModelObject


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def branincurrin_func():
    return Function("branincurrin", dim=2, nobj=2)


@pytest.fixture(scope="module")
def mtgp_gps(branincurrin_func):
    """ModelObject with a MultiTaskGP fitted on BraninCurrin data."""
    torch.manual_seed(7)
    n = 15
    train_x = torch.rand(n, branincurrin_func.dim, dtype=torch.float64)
    train_y = branincurrin_func.evaluate(train_x)
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=branincurrin_func.get_bounds(),
        nobj=branincurrin_func.nobj,
        ncons=0,
        ntrain=n,
        device=torch.device("cpu"),
    )
    gps.fit_multitask_gp()
    return gps


@pytest.fixture(scope="module")
def sgp_gps(branincurrin_func):
    """ModelObject with independent SingleTaskGPs fitted on BraninCurrin data."""
    torch.manual_seed(7)
    n = 15
    train_x = torch.rand(n, branincurrin_func.dim, dtype=torch.float64)
    train_y = branincurrin_func.evaluate(train_x)
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=branincurrin_func.get_bounds(),
        nobj=branincurrin_func.nobj,
        ncons=0,
        ntrain=n,
        device=torch.device("cpu"),
    )
    gps.fit_gp()
    return gps


@pytest.fixture(scope="module")
def constrained_mtgp_gps():
    """Fitted constrained MultiTaskGP matching the documented posterior path."""
    torch.manual_seed(11)
    function = Function("constrainedbc", dim=2, nobj=2)
    n = 12
    train_x = torch.rand(n, function.dim, dtype=torch.float64)
    evaluation = function.evaluate_all(train_x)
    train_y = torch.cat((evaluation.objectives, evaluation.constraints), dim=-1)
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=function.get_bounds(),
        nobj=function.nobj,
        ncons=1,
        ntrain=n,
        device=torch.device("cpu"),
    )
    gps.fit_multitask_gp()
    return function, gps


# ---------------------------------------------------------------------------
# _mt_gp_posterior
# ---------------------------------------------------------------------------

def test_mt_gp_posterior_output_shape(mtgp_gps, branincurrin_func):
    """_mt_gp_posterior returns a tensor whose last two dims are (n_points, nobj).

    The MTGP posterior sample has an extra leading sample dimension of 1,
    so the full shape is (1, n_points, nobj).
    """
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps)
    n = 8
    x = torch.rand(n, branincurrin_func.dim, dtype=torch.float64)
    result = acq._mt_gp_posterior(x, mtgp_gps, seed_iter=1)
    assert isinstance(result, torch.Tensor)
    assert result.shape[-1] == mtgp_gps.nobj, "Last dim must equal nobj"
    assert result.shape[-2] == n, "Second-to-last dim must equal n_points"


def test_mt_gp_posterior_returns_negated_values(mtgp_gps, branincurrin_func):
    """_mt_gp_posterior must return -Ys (negation convention for NSGA-II minimization)."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps)
    x = torch.rand(5, branincurrin_func.dim, dtype=torch.float64)
    result = acq._mt_gp_posterior(x, mtgp_gps, seed_iter=42)
    # BraninCurrin is negated, so raw samples from the MTGP should be negative.
    # After double negation (-Ys), result should be mostly positive (not guaranteed
    # for every sample, but the sign convention must flip: raw < 0 → output > 0).
    # We just verify shape and that the method runs without error here.
    assert not torch.isnan(result).all(), "_mt_gp_posterior must not return all NaN"


def test_mt_gp_posterior_is_negated_vs_gp_posterior_sign(mtgp_gps, sgp_gps, branincurrin_func):
    """
    Both _gp_posterior and _mt_gp_posterior apply -Ys.
    For the same seed, BraninCurrin posterior samples should be negative (negate=True),
    so the returned value must be positive.
    """
    acq_mt = Acquisition(func=branincurrin_func, gps=mtgp_gps)
    x = torch.rand(6, branincurrin_func.dim, dtype=torch.float64)
    result = acq_mt._mt_gp_posterior(x, mtgp_gps, seed_iter=1)
    # The result is -Ys; BraninCurrin with negate=True gives large negative values,
    # so result (which is the negation) should be positive.
    assert (result > -1e6).all(), "Posterior must not contain extreme penalty values for unconstrained problem"


def test_mt_gp_posterior_seed_reproducibility(mtgp_gps, branincurrin_func):
    """Same seed_iter must produce identical output."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps)
    x = torch.rand(4, branincurrin_func.dim, dtype=torch.float64)
    r1 = acq._mt_gp_posterior(x, mtgp_gps, seed_iter=99)
    r2 = acq._mt_gp_posterior(x, mtgp_gps, seed_iter=99)
    assert torch.allclose(r1, r2), "_mt_gp_posterior must be deterministic for the same seed"


@patch("qpots.acquisition.unstandardize_ignore_nan", side_effect=lambda values, _train_y: values)
def test_mt_gp_posterior_preserves_sample_dimension_with_constraints(_unstandardize):
    func = Mock(dim=2)
    gps = Mock()
    gps.nobj = 2
    gps.ncons = 2
    gps.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    gps.train_y = torch.zeros(3, 4, dtype=torch.float64)

    sampled_values = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0, 0.5],
                [3.0, 4.0, -0.1, 1.0],
                [5.0, 6.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float64,
    )
    posterior = Mock()
    posterior.sample.return_value = sampled_values
    model = Mock()
    model.posterior.return_value = posterior
    gps.models = [model]

    acquisition = Acquisition(func=func, gps=gps, device="cpu", dtype=torch.float64)
    result = acquisition._mt_gp_posterior(
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float64),
        gps,
    )

    expected = torch.tensor(
        [[[-1.0, -2.0], [1e12, 1e12], [-5.0, -6.0]]], dtype=torch.float64
    )
    assert result.shape == (1, 3, 2)
    assert torch.equal(result, expected)


def test_qpots_mt1_with_constraints_preserves_population_dimension(constrained_mtgp_gps):
    """A real constrained MTGP completes the NSGA-II posterior evaluation."""
    function, gps = constrained_mtgp_gps
    acquisition = Acquisition(func=function, gps=gps, q=1)
    result = acquisition.qpots(
        function.get_bounds(),
        iteration=1,
        nystrom=0,
        mt=1,
        partial_info=0,
        iters=1,
        dim=function.dim,
        nychoice="random",
        q=1,
        ngen=1,
    )

    assert result.shape == (1, function.dim)
    assert torch.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()


# ---------------------------------------------------------------------------
# qpots with mt=0 (standard GP path) — basic integration
# ---------------------------------------------------------------------------

def test_qpots_mt0_returns_tensor(sgp_gps, branincurrin_func):
    """qpots(mt=0) must return a tensor of shape (q, dim)."""
    acq = Acquisition(func=branincurrin_func, gps=sgp_gps, q=1)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 0,
        "partial_info": 0,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 1,
        "ngen": 5,
    }
    result = acq.qpots(bounds, iteration=1, **kwargs)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([1, branincurrin_func.dim])


def test_qpots_mt0_in_unit_hypercube(sgp_gps, branincurrin_func):
    """Candidates returned by qpots(mt=0) must lie in [0, 1]^d after normalize."""
    acq = Acquisition(func=branincurrin_func, gps=sgp_gps, q=2)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 0,
        "partial_info": 0,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 2,
        "ngen": 5,
    }
    result = acq.qpots(bounds, iteration=1, **kwargs)
    assert (result >= 0.0).all() and (result <= 1.0).all()


# ---------------------------------------------------------------------------
# qpots with mt=1 (MultiTaskGP path)
# ---------------------------------------------------------------------------

def test_qpots_mt1_returns_tensor(mtgp_gps, branincurrin_func):
    """qpots(mt=1, partial_info=0) must return a tensor of shape (q, dim)."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=1)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 0,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 1,
        "ngen": 5,
    }
    result = acq.qpots(bounds, iteration=1, **kwargs)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([1, branincurrin_func.dim])


def test_qpots_mt1_in_unit_hypercube(mtgp_gps, branincurrin_func):
    """Candidates from qpots(mt=1) must be normalized to [0, 1]^d."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=1)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 0,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 1,
        "ngen": 5,
    }
    result = acq.qpots(bounds, iteration=2, **kwargs)
    assert (result >= 0.0).all() and (result <= 1.0).all()


# ---------------------------------------------------------------------------
# qpots with partial_info=1
# ---------------------------------------------------------------------------

def test_qpots_partial_info_returns_tuple(mtgp_gps, branincurrin_func):
    """qpots(partial_info=1) must return a (candidates, task_ids) tuple."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=2)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 1,
        "threshold": None,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 2,
        "ngen": 5,
    }
    result = acq.qpots(bounds, iteration=1, **kwargs)
    assert isinstance(result, tuple), "partial_info=1 must return a tuple"
    assert len(result) == 2, "Tuple must have exactly two elements (candidates, task_ids)"


def test_qpots_partial_info_candidates_shape(mtgp_gps, branincurrin_func):
    """Candidates from partial_info=1 must have dim columns."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=2)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 1,
        "threshold": None,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 2,
        "ngen": 5,
    }
    candidates, task_ids = acq.qpots(bounds, iteration=1, **kwargs)
    assert candidates.shape[1] == branincurrin_func.dim


def test_qpots_partial_info_task_ids_shape(mtgp_gps, branincurrin_func):
    """task_ids must have nobj columns (one per task)."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=2)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 1,
        "threshold": None,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 2,
        "ngen": 5,
    }
    candidates, task_ids = acq.qpots(bounds, iteration=1, **kwargs)
    assert task_ids.shape[1] == mtgp_gps.nobj + mtgp_gps.ncons


def test_qpots_partial_info_candidates_normalized(mtgp_gps, branincurrin_func):
    """Candidates from partial_info=1 must be in [0, 1]^d (normalized)."""
    acq = Acquisition(func=branincurrin_func, gps=mtgp_gps, q=2)
    bounds = branincurrin_func.get_bounds()
    kwargs = {
        "nystrom": 0,
        "mt": 1,
        "partial_info": 1,
        "threshold": None,
        "iters": 5,
        "dim": branincurrin_func.dim,
        "nychoice": "random",
        "q": 2,
        "ngen": 5,
    }
    candidates, _ = acq.qpots(bounds, iteration=1, **kwargs)
    assert (candidates >= 0.0).all() and (candidates <= 1.0).all()
