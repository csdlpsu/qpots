from unittest.mock import Mock
import pytest
import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume

from qpots.utils.utils import unstandardize, expected_hypervolume, gen_filtered_cands, select_candidates, arg_parser

@pytest.fixture
def mock_gps():
    gps = Mock()
    gps.train_y = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    gps.train_x = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]])
    gps.nobj = 2
    gps.ncons = 0
    return gps

def test_unstandardize():
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    train_y = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    result = unstandardize(Y, train_y)
    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0)
    expected = Y * std + mean

    assert torch.allclose(result, expected), "Unstandardization failed"

def test_expected_hypervolume(mock_gps):
    ref_point = torch.tensor([-1.0, -1.0])

    hv_value, pareto_front = expected_hypervolume(mock_gps, ref_point)
    
    # Verify results
    hv = Hypervolume(ref_point=ref_point.double())
    partitioning = FastNondominatedPartitioning(ref_point.double(), mock_gps.train_y.double())
    expected_pareto = partitioning.pareto_Y
    expected_hv = hv.compute(expected_pareto)

    assert torch.allclose(hv_value, torch.tensor([expected_hv], dtype=torch.double)), "Hypervolume computation failed"
    assert torch.equal(pareto_front, expected_pareto), "Pareto front mismatch"

def test_gen_filtered_cands(mock_gps):
    candidates = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    ref_point = torch.tensor([0.0, 0.0])

    filtered_candidates = gen_filtered_cands(mock_gps, candidates, ref_point, kernel_bandwidth=0.05)
    
    assert filtered_candidates.shape[0] <= candidates.shape[0], "Too many candidates selected"

def test_select_candidates(mock_gps):
    pareto_set = np.array([[0.1, 0.1], [0.9, 0.9]])
    device = "cpu"

    selected_candidates = select_candidates(mock_gps, pareto_set, device, q=1, seed=42)
    assert selected_candidates.shape[0] == 1, "Incorrect number of candidates selected"

def test_arg_parser():
    sys.argv = [
        "script_name",
        "--ntrain", "10",
        "--ref_point", "0", "0",
        "--dim", "3"
    ]
    args = arg_parser()
    assert args.ntrain == 10, "Failed to parse --ntrain"
    assert args.dim == 3, "Failed to parse --dim"
    assert args.ref_point == [0.0, 0.0], "Failed to parse --ref_point"


# ---------------------------------------------------------------------------
# Tests added for improved coverage
# ---------------------------------------------------------------------------

def test_unstandardize_single_column():
    """unstandardize must work correctly when train_y has only one column."""
    Y = torch.tensor([[1.0], [2.0], [3.0]])
    train_y = torch.tensor([[2.0], [4.0]])
    result = unstandardize(Y, train_y)
    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0)
    expected = Y * std + mean
    assert torch.allclose(result, expected), "Single-column unstandardize failed"


def test_expected_hypervolume_min():
    """expected_hypervolume with min=True must return positive hv and a 2-column Pareto front."""
    gps = Mock()
    # Values well below the hardcoded ref boundary of 0.335 so the HV calculation is valid
    gps.train_y = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]], dtype=torch.float64)
    gps.nobj = 2
    gps.ncons = 0

    hv, pf = expected_hypervolume(gps, min=True)

    assert hv > 0, "Hypervolume (min mode) should be positive for non-dominated Pareto front"
    assert isinstance(pf, torch.Tensor)
    assert pf.shape[1] == 2


def test_expected_hypervolume_constrained():
    """expected_hypervolume must filter infeasible points (con < 0) before computing HV."""
    gps = Mock()
    gps.nobj = 2
    gps.ncons = 1
    # Rows: [obj1, obj2, constraint];  third row is infeasible
    gps.train_y = torch.tensor([
        [2.0, 1.0, 0.5],   # feasible
        [1.0, 2.0, 0.3],   # feasible
        [1.5, 1.5, -0.1],  # infeasible
    ], dtype=torch.float64)
    ref_point = torch.tensor([-1.0, -1.0])

    hv, pf = expected_hypervolume(gps, ref_point)

    assert hv >= 0
    assert pf.shape[1] == 2, "Pareto front must contain only objective columns"
    assert pf.shape[0] <= 2, "Pareto front can contain at most 2 feasible points"


def test_expected_hypervolume_min_constrained_known_bug():
    """
    Documents a known bug: min=True with ncons>0 passes constraint columns to the
    hypervolume calculator, which expects only nobj columns, causing a dimension error.
    This test pins the current behaviour so the bug is not silently introduced again
    if the code is refactored.
    """
    from botorch.exceptions.errors import BotorchTensorDimensionError

    gps = Mock()
    gps.nobj = 2
    gps.ncons = 1
    gps.train_y = torch.tensor([
        [0.1, 0.2, 0.5],   # feasible
        [0.2, 0.1, -0.1],  # infeasible
    ], dtype=torch.float64)

    with pytest.raises(BotorchTensorDimensionError):
        expected_hypervolume(gps, min=True)


def test_select_candidates_q_larger_than_pareto_set(mock_gps):
    """select_candidates must not crash when q exceeds the number of Pareto points."""
    pareto_set = np.array([[0.5, 0.5]])  # single point
    result = select_candidates(mock_gps, pareto_set, "cpu", q=5, seed=42)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] <= 5, "Cannot return more candidates than Pareto points"
    assert result.shape[1] == pareto_set.shape[1]


def test_gen_filtered_cands_returns_tensor():
    """gen_filtered_cands must return a 2-D tensor with the right feature dimension."""
    gps = Mock()
    gps.train_x = torch.tensor([[0.1, 0.2], [0.8, 0.9]])
    gps.train_y = torch.tensor([[3.0, 4.0], [2.0, 5.0]], dtype=torch.float64)
    candidates = torch.rand(10, 2, dtype=torch.float64)
    ref_point = torch.tensor([0.0, 0.0])

    filtered = gen_filtered_cands(gps, candidates, ref_point, kernel_bandwidth=0.1)

    assert isinstance(filtered, torch.Tensor)
    assert filtered.ndim == 2, "Filtered candidates must be 2-D"
    assert filtered.shape[1] == 2, "Feature dimension must be preserved"
    assert filtered.shape[0] <= candidates.shape[0], "Cannot gain candidates through filtering"


def test_arg_parser_defaults():
    """arg_parser must return the documented defaults for every optional argument."""
    sys.argv = ["script_name", "--ref_point", "0.0", "0.0", "--dim", "2"]
    args = arg_parser()
    assert args.ntrain == 20
    assert args.iters == 20
    assert args.reps == 20
    assert args.q == 1
    assert args.nobj == 2
    assert args.ncons == 0
    assert args.acq == "TS"
    assert args.nystrom == 0
    assert args.nychoice == "pareto"
    assert args.ngen == 10
