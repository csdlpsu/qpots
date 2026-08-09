from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from qpots.tsemo_runner import TSEMORunner


@pytest.fixture
def tsemo_path(tmp_path):
    """Create the directory shape expected from an external TS-EMO checkout."""
    (tmp_path / "TSEMO_run.m").touch()
    for relative in (
        "Test_functions",
        "Direct",
        "Mex_files/invchol",
        "Mex_files/hypervolume",
        "Mex_files/pareto front",
        "NGPM_v1.4",
    ):
        (tmp_path / relative).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def tsemo_runner(tsemo_path):
    with patch("matlab.engine.start_matlab", return_value=Mock()):
        return TSEMORunner(
            func="test_function",
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            y=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            lb=[0, 0],
            ub=[1, 1],
            iters=5,
            batch_number=2,
            tsemo_path=tsemo_path,
        )


def test_tsemo_path_is_required():
    with pytest.raises(ValueError, match="no longer redistributes"):
        TSEMORunner("test", [], [], [], [], 1, 1)


def test_tsemo_path_is_validated_before_matlab_starts(tmp_path):
    with patch("matlab.engine.start_matlab") as start_matlab:
        with pytest.raises(FileNotFoundError, match="TSEMO_run.m"):
            TSEMORunner("test", [], [], [], [], 1, 1, tsemo_path=tmp_path)
    start_matlab.assert_not_called()


def test_tsemo_runner_init(tsemo_runner, tsemo_path):
    assert tsemo_runner._func == "test_function"
    assert tsemo_runner._tsemo_path == tsemo_path.resolve()
    assert tsemo_runner._iters == 5
    assert tsemo_runner._batch_number == 2
    assert isinstance(tsemo_runner._eng, Mock)
    assert tsemo_runner._eng.addpath.call_count == 7


def test_tsemo_run(tsemo_runner, tmp_path):
    mock_x = [[0.5, 0.6], [0.7, 0.8]]
    mock_y = [[5.0, 6.0], [7.0, 8.0]]
    mock_times = [0.1, 0.2]
    tsemo_runner._eng.TSEMO_run = MagicMock(return_value=(mock_x, mock_y, mock_times))

    x, y, times = tsemo_runner.tsemo_run(tmp_path, rep=1)

    tsemo_runner._eng.TSEMO_run.assert_called_once()
    np.testing.assert_array_equal(x, np.array(mock_x))
    np.testing.assert_array_equal(y, np.array(mock_y))
    np.testing.assert_array_equal(times, np.array(mock_times))
    np.testing.assert_array_equal(np.load(tmp_path / "X_1.npy"), np.array(mock_x))
    np.testing.assert_array_equal(np.load(tmp_path / "Y_1.npy"), np.array(mock_y))
    np.testing.assert_array_equal(np.load(tmp_path / "times_1.npy"), np.array(mock_times))


@pytest.mark.parametrize("iterations", [2, 4])
def test_tsemo_hypervolume_is_non_decreasing(tsemo_runner, iterations):
    outcomes = torch.tensor(
        [
            [-1.0, -2.0],
            [-2.0, -1.5],
            [-1.5, -1.8],
            [-3.0, -2.5],
            [-2.8, -2.0],
            [-1.2, -3.0],
        ],
        dtype=torch.float64,
    )
    hypervolume, pareto_front = tsemo_runner.tsemo_hypervolume(
        outcomes,
        ref_point=torch.tensor([-4.0, -4.0]),
        train_shape=2,
        iters=iterations,
    )

    assert len(hypervolume) == iterations
    assert all(float(value) >= 0 for value in hypervolume)
    assert all(
        float(current) >= float(previous) - 1e-8
        for previous, current in zip(hypervolume, hypervolume[1:], strict=False)
    )
    assert pareto_front.shape[1] == 2
