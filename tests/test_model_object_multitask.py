import pytest
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.models import MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from qpots.model_object import ModelObject


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_gp():
    """Minimal ModelObject with 10 fully-observed training points, 2 objectives."""
    torch.manual_seed(0)
    train_x = torch.rand(10, 2, dtype=torch.float64)
    train_y = torch.rand(10, 2, dtype=torch.float64)
    bounds = torch.zeros(2, 2, dtype=torch.float64)
    bounds[1] = 1.0
    return ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        nobj=2,
        ncons=0,
        ntrain=10,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def partial_gp():
    """ModelObject where the last 3 rows have NaN in one or both objectives."""
    torch.manual_seed(1)
    n_full, n_partial, dim = 10, 3, 2
    train_x = torch.rand(n_full + n_partial, dim, dtype=torch.float64)
    # Full observations for the first n_full rows
    train_y_full = torch.rand(n_full, 2, dtype=torch.float64)
    # Partial observations: obj0 observed, obj1 is NaN
    train_y_partial = torch.stack(
        [torch.rand(n_partial, dtype=torch.float64), torch.full((n_partial,), float("nan"))],
        dim=-1,
    )
    train_y = torch.cat([train_y_full, train_y_partial], dim=0)
    bounds = torch.zeros(2, dim, dtype=torch.float64)
    bounds[1] = 1.0
    return ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        nobj=2,
        ncons=0,
        ntrain=n_full,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# fit_multitask_gp — no NaN
# ---------------------------------------------------------------------------

def test_fit_multitask_gp_produces_one_model(base_gp):
    base_gp.fit_multitask_gp()
    assert len(base_gp.models) == 1, "fit_multitask_gp must append exactly one model"
    assert len(base_gp.mlls) == 1


def test_fit_multitask_gp_model_type(base_gp):
    base_gp.fit_multitask_gp()
    assert isinstance(base_gp.models[0], MultiTaskGP)


def test_fit_multitask_gp_mll_type(base_gp):
    base_gp.fit_multitask_gp()
    assert isinstance(base_gp.mlls[0], ExactMarginalLogLikelihood)


def test_fit_multitask_gp_posterior_shape(base_gp):
    """MTGP posterior at test points must have the right shape."""
    base_gp.fit_multitask_gp()
    model = base_gp.models[0]
    # Build a single test point with task_id appended
    x_test = torch.cat(
        [torch.rand(1, 2, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)],
        dim=-1,
    )
    posterior = model.posterior(x_test)
    assert posterior.mean.shape[0] == 1


# ---------------------------------------------------------------------------
# fit_multitask_gp — with partial NaN (partial information)
# ---------------------------------------------------------------------------

def test_fit_multitask_gp_with_nan_completes(partial_gp):
    """fit_multitask_gp must not raise when train_y contains NaN rows."""
    partial_gp.fit_multitask_gp()
    assert len(partial_gp.models) == 1


def test_fit_multitask_gp_with_nan_model_is_mtgp(partial_gp):
    partial_gp.fit_multitask_gp()
    assert isinstance(partial_gp.models[0], MultiTaskGP)


# ---------------------------------------------------------------------------
# standardize_ignore_nan
# ---------------------------------------------------------------------------

def test_standardize_ignore_nan_no_nan(base_gp):
    """Without NaN, standardize_ignore_nan must produce zero mean and unit std per column."""
    Y = torch.rand(20, 3, dtype=torch.float64)
    Y_std = base_gp.standardize_ignore_nan(Y)
    for col in range(3):
        col_vals = Y_std[:, col]
        assert not torch.isnan(col_vals).any()
        assert abs(col_vals.mean().item()) < 1e-9, "Column mean must be ~0 after standardization"
        assert abs(col_vals.std().item() - 1.0) < 1e-6, "Column std must be ~1 after standardization"


def test_standardize_ignore_nan_preserves_nan_positions(base_gp):
    """NaN entries in Y must remain NaN after standardization."""
    Y = torch.tensor([[1.0, float("nan")], [2.0, 3.0], [3.0, 5.0]], dtype=torch.float64)
    Y_std = base_gp.standardize_ignore_nan(Y)
    assert torch.isnan(Y_std[0, 1]), "NaN must stay NaN after standardize_ignore_nan"
    assert not torch.isnan(Y_std[1, 1])


def test_standardize_ignore_nan_non_nan_columns_correct(base_gp):
    """Columns without NaN must be exactly standardized."""
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    Y_std = base_gp.standardize_ignore_nan(Y)
    for col in range(2):
        assert abs(Y_std[:, col].mean().item()) < 1e-9
        assert abs(Y_std[:, col].std().item() - 1.0) < 1e-6


def test_standardize_ignore_nan_constant_column(base_gp):
    """A column with all identical (non-NaN) values must not produce NaN (std=0 guard)."""
    Y = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]], dtype=torch.float64)
    Y_std = base_gp.standardize_ignore_nan(Y)
    assert not torch.isnan(Y_std[:, 0]).any(), "Constant column should not produce NaN"


# ---------------------------------------------------------------------------
# ntrain update inside fit_gp
# ---------------------------------------------------------------------------

def test_ntrain_updated_by_fit_gp(base_gp):
    """fit_gp must set ntrain to the current number of training points."""
    initial_ntrain = 5  # intentionally wrong
    base_gp.ntrain = initial_ntrain
    base_gp.fit_gp()
    assert base_gp.ntrain == base_gp.train_x.shape[0], (
        "fit_gp must update ntrain to train_x.shape[0]"
    )
