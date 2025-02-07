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
