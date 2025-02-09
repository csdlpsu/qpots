import pytest
import torch
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpots.tsemo_runner import TSEMORunner

@pytest.fixture
def tsemo_runner():
    with patch("matlab.engine.start_matlab", return_value=Mock()) as mock_engine:
        runner = TSEMORunner(
            func="test_function",
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            y=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            lb=[0, 0],
            ub=[1, 1],
            iters=5,
            batch_number=2
        )
        return runner

def test_tsemo_runner_init(tsemo_runner):
    assert tsemo_runner._func == "test_function"
    assert len(tsemo_runner._x) == 2
    assert len(tsemo_runner._y) == 2
    assert tsemo_runner._iters == 5
    assert tsemo_runner._batch_number == 2
    assert isinstance(tsemo_runner._eng, Mock)  # MATLAB engine should be mocked

def test_tsemo_run(tsemo_runner, tmp_path):
    mock_X_out = [[0.5, 0.6], [0.7, 0.8]]
    mock_Y_out = [[5.0, 6.0], [7.0, 8.0]]
    mock_times = [0.1, 0.2]

    tsemo_runner._eng.TSEMO_run = MagicMock(return_value=(mock_X_out, mock_Y_out, mock_times))

    save_dir = tmp_path  # Temporary directory for testing file saving
    rep = 1

    X, Y, times_np = tsemo_runner.tsemo_run(save_dir, rep)

    # Ensure the MATLAB function was called correctly
    tsemo_runner._eng.TSEMO_run.assert_called_once()

    # Verify returned values
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(times_np, np.ndarray)
    assert X.shape == (2, 2)
    assert Y.shape == (2, 2)

    # Check that files were saved correctly
    assert (save_dir / "X_1.npy").exists()
    assert (save_dir / "Y_1.npy").exists()
    assert (save_dir / "times_1.npy").exists()

    # Check file contents
    np.testing.assert_array_equal(np.load(save_dir / "X_1.npy"), np.array(mock_X_out))
    np.testing.assert_array_equal(np.load(save_dir / "Y_1.npy"), np.array(mock_Y_out))
    np.testing.assert_array_equal(np.load(save_dir / "times_1.npy"), np.array(mock_times))

def test_tsemo_hypervolume():
    Y = torch.tensor([[1.0, 2.0], [2.0, 1.5], [1.5, 1.8], [3.0, 2.5]])
    ref_point = torch.tensor([4.0, 4.0])
    train_shape = 2
    iters = 2

    tsemo_runner = TSEMORunner(
        func="test_function",
        x=[],
        y=[],
        lb=[],
        ub=[],
        iters=iters,
        batch_number=1
    )

    hv, pf = tsemo_runner.tsemo_hypervolume(Y, ref_point, train_shape, iters)

    assert isinstance(hv, list), "Hypervolume should be a list"
    assert len(hv) == iters, "Hypervolume list length mismatch"
    assert isinstance(pf, torch.Tensor), "Pareto front should be a tensor"
    assert pf.shape[1] == 2, "Pareto front shape mismatch"
