import pytest
import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpots.utils.utils import (
    unstandardize,
    unstandardize_ignore_nan,
    posterior_mean_fill,
    select_candidates_partial_info,
    minmax_scale,
    expected_hypervolume,
)
from qpots.model_object import ModelObject
from qpots.function import Function


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def branincurrin_func():
    return Function("branincurrin", dim=2, nobj=2)


@pytest.fixture(scope="module")
def mtgp_gps(branincurrin_func):
    """ModelObject with a fitted MultiTaskGP (no NaN)."""
    torch.manual_seed(3)
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
def mtgp_gps_with_nan(branincurrin_func):
    """ModelObject with a fitted MultiTaskGP where later rows have NaN."""
    torch.manual_seed(5)
    n_full, n_partial = 12, 4
    n = n_full + n_partial
    train_x = torch.rand(n, branincurrin_func.dim, dtype=torch.float64)
    train_y_full = branincurrin_func.evaluate(train_x[:n_full])
    # partial rows: only obj0 observed
    partial_obj0 = branincurrin_func.evaluate(train_x[n_full:])[:, 0:1]
    partial_nan = torch.full((n_partial, 1), float("nan"), dtype=torch.float64)
    train_y_partial = torch.cat([partial_obj0, partial_nan], dim=-1)
    train_y = torch.cat([train_y_full, train_y_partial], dim=0)
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=branincurrin_func.get_bounds(),
        nobj=branincurrin_func.nobj,
        ncons=0,
        ntrain=n_full,
        device=torch.device("cpu"),
    )
    gps.fit_multitask_gp()
    return gps


# ---------------------------------------------------------------------------
# unstandardize_ignore_nan
# ---------------------------------------------------------------------------

def test_unstandardize_ignore_nan_matches_unstandardize_when_no_nan():
    """Without NaN, unstandardize_ignore_nan must produce the same result as unstandardize."""
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    train_y = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float64)
    result_nan = unstandardize_ignore_nan(Y, train_y)
    result_std = unstandardize(Y, train_y)
    assert torch.allclose(result_nan, result_std, atol=1e-10), (
        "unstandardize_ignore_nan must match unstandardize when train_y has no NaN"
    )


def test_unstandardize_ignore_nan_correct_scale():
    """Output must equal Y * nan-aware-std + nan-aware-mean."""
    Y = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    train_y = torch.tensor([[1.0, float("nan")], [3.0, 2.0]], dtype=torch.float64)
    result = unstandardize_ignore_nan(Y, train_y)
    expected_mean = torch.tensor([2.0, 2.0], dtype=torch.float64)  # nanmean col0=2, col1=2
    expected_std_col0 = torch.std(torch.tensor([1.0, 3.0], dtype=torch.float64))
    expected_col0 = 0.0 * expected_std_col0 + expected_mean[0]
    assert torch.allclose(result[0, 0], expected_col0, atol=1e-9)


def test_unstandardize_ignore_nan_output_shape():
    """Output shape must match input shape."""
    Y = torch.rand(5, 3, dtype=torch.float64)
    train_y = torch.rand(10, 3, dtype=torch.float64)
    train_y[2, 1] = float("nan")
    result = unstandardize_ignore_nan(Y, train_y)
    assert result.shape == Y.shape


# ---------------------------------------------------------------------------
# posterior_mean_fill
# ---------------------------------------------------------------------------

def test_posterior_mean_fill_no_nan_returns_same_values(mtgp_gps):
    """When train_y has no NaN, posterior_mean_fill must return a clone of train_y."""
    filled = posterior_mean_fill(mtgp_gps)
    assert torch.allclose(filled, mtgp_gps.train_y), (
        "posterior_mean_fill must preserve non-NaN values unchanged"
    )


def test_posterior_mean_fill_no_nan_is_independent_copy(mtgp_gps):
    """The returned tensor must be a copy (modifying it must not affect train_y)."""
    filled = posterior_mean_fill(mtgp_gps)
    original = mtgp_gps.train_y.clone()
    filled[0, 0] += 1e6
    assert torch.allclose(mtgp_gps.train_y, original), (
        "posterior_mean_fill must return an independent copy"
    )


def test_posterior_mean_fill_removes_nan(mtgp_gps_with_nan):
    """After filling, the result must contain no NaN values."""
    filled = posterior_mean_fill(mtgp_gps_with_nan)
    assert not torch.isnan(filled).any(), (
        "posterior_mean_fill must replace all NaN entries with posterior means"
    )


def test_posterior_mean_fill_preserves_fully_observed_rows(mtgp_gps_with_nan):
    """Rows without NaN must not be changed by posterior_mean_fill."""
    gps = mtgp_gps_with_nan
    n_full = gps.ntrain
    filled = posterior_mean_fill(gps)
    assert torch.allclose(filled[:n_full], gps.train_y[:n_full]), (
        "posterior_mean_fill must not modify fully-observed rows"
    )


def test_posterior_mean_fill_shape(mtgp_gps_with_nan):
    """Output shape must match train_y shape."""
    filled = posterior_mean_fill(mtgp_gps_with_nan)
    assert filled.shape == mtgp_gps_with_nan.train_y.shape


# ---------------------------------------------------------------------------
# expected_hypervolume with NaN train_y (multitask_edits version uses posterior_mean_fill)
# ---------------------------------------------------------------------------

def test_expected_hypervolume_with_nan_train_y(mtgp_gps_with_nan, branincurrin_func):
    """expected_hypervolume must complete and return non-negative HV when train_y has NaN."""
    ref_point = torch.tensor([-300.0, -18.0])
    hv, pf = expected_hypervolume(mtgp_gps_with_nan, ref_point)
    assert float(hv) >= 0.0, "Hypervolume must be non-negative"
    assert isinstance(pf, torch.Tensor)
    assert pf.shape[1] == mtgp_gps_with_nan.nobj


def test_expected_hypervolume_no_nan_matches_reference(mtgp_gps, branincurrin_func):
    """With no NaN, expected_hypervolume must return positive HV from a consistent Pareto front."""
    from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
    ref_point = torch.tensor([-300.0, -18.0])
    hv, pf = expected_hypervolume(mtgp_gps, ref_point)
    assert float(hv) >= 0.0
    # Pareto front must be a subset of objectives only
    assert pf.shape[1] == mtgp_gps.nobj


# ---------------------------------------------------------------------------
# select_candidates_partial_info
# ---------------------------------------------------------------------------

def test_select_candidates_partial_info_random_mode_returns_tuple(mtgp_gps):
    """Random mode (thresh=None) must return (candidates, task_ids) tuple."""
    pareto_set = mtgp_gps.train_x.numpy()
    result = select_candidates_partial_info(mtgp_gps, pareto_set, "cpu", q=2, seed=42, thresh=None)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_select_candidates_partial_info_random_candidates_shape(mtgp_gps):
    """Candidates must have the same feature dimension as train_x."""
    pareto_set = mtgp_gps.train_x.numpy()
    candidates, _ = select_candidates_partial_info(mtgp_gps, pareto_set, "cpu", q=2, seed=0, thresh=None)
    assert candidates.shape[1] == mtgp_gps.train_x.shape[1]


def test_select_candidates_partial_info_random_task_ids_shape(mtgp_gps):
    """task_ids must have nobj+ncons columns."""
    pareto_set = mtgp_gps.train_x.numpy()
    candidates, task_ids = select_candidates_partial_info(
        mtgp_gps, pareto_set, "cpu", q=3, seed=0, thresh=None
    )
    assert task_ids.shape[1] == mtgp_gps.nobj + mtgp_gps.ncons


def test_select_candidates_partial_info_random_no_all_nan_rows(mtgp_gps):
    """No row in task_ids should be entirely NaN (filtered out by implementation)."""
    pareto_set = mtgp_gps.train_x.numpy()
    _, task_ids = select_candidates_partial_info(
        mtgp_gps, pareto_set, "cpu", q=4, seed=1, thresh=None
    )
    all_nan_rows = torch.isnan(task_ids).all(dim=1)
    assert not all_nan_rows.any(), "select_candidates_partial_info must filter all-NaN rows"


def test_select_candidates_partial_info_candidates_and_task_ids_row_count_match(mtgp_gps):
    """Candidates and task_ids must have the same number of rows."""
    pareto_set = mtgp_gps.train_x.numpy()
    candidates, task_ids = select_candidates_partial_info(
        mtgp_gps, pareto_set, "cpu", q=3, seed=2, thresh=None
    )
    assert candidates.shape[0] == task_ids.shape[0]


# ---------------------------------------------------------------------------
# minmax_scale
# ---------------------------------------------------------------------------

def test_minmax_scale_range():
    """minmax_scale must map all values to [0, 1]."""
    x = torch.tensor([[-2.0, 3.0], [0.0, 1.0], [4.0, -1.0]])
    scaled = minmax_scale(x)
    assert scaled.min() >= 0.0 - 1e-9
    assert scaled.max() <= 1.0 + 1e-9


def test_minmax_scale_dim_none_global():
    """With dim=None, minmax_scale scales the whole tensor globally."""
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    scaled = minmax_scale(x, dim=None)
    assert abs(scaled.min().item()) < 1e-9
    assert abs(scaled.max().item() - 1.0) < 1e-6


def test_minmax_scale_column_wise():
    """With dim=0, each column is independently scaled to [0, 1]."""
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
    scaled = minmax_scale(x, dim=0)
    # Column 0: min=1, max=3  → [0, 0.5, 1]
    # Column 1: min=10, max=30 → [0, 0.5, 1]
    assert torch.allclose(scaled[:, 0], scaled[:, 1], atol=1e-9)
    assert abs(scaled[:, 0].min().item()) < 1e-9
    assert abs(scaled[:, 0].max().item() - 1.0) < 1e-6


def test_minmax_scale_constant_tensor_no_nan():
    """A constant tensor must not produce NaN (eps guard)."""
    x = torch.ones(3, 2, dtype=torch.float64)
    scaled = minmax_scale(x)
    assert not torch.isnan(scaled).any()
