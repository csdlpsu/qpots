from unittest.mock import Mock, patch
import pytest
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.utils.sampling import manual_seed
from qpots.acquisition import Acquisition
from qpots.function import Function
from qpots.model_object import ModelObject

@pytest.fixture
def mock_gp_no_cons():
    gps = Mock()
    gps.nobj = 2
    gps.ncons = 0
    gps.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    gps.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    gps.train_y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    return gps

@pytest.fixture()
def mock_gp_cons():
    gps = Mock()
    gps.nobj = 2
    gps.ncons = 1
    gps.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    gps.train_x = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    gps.train_y = torch.tensor([[1.0, 5.0, -0.5], [2.0, 3.0, 0.3]])
    return gps

@pytest.fixture
def mock_func():
    func = Mock()
    func.name = "mock_function"
    func.dim = 2
    return func

@pytest.fixture
def real_func():
    func = Function('branincurrin', dim=2, nobj=2)
    return func

def test_acquisition_init(mock_gp_cons, mock_func):
    cons = Mock()

    # Parameters
    device = torch.device("cpu")
    q = 5
    num_restarts = 20
    raw_samples = 1024

    # Instantiate Acquisition
    acquisition = Acquisition(
        func=mock_func,
        gps=mock_gp_cons,
        cons=cons,
        device=device,
        q=q,
        NUM_RESTARTS=num_restarts,
        RAW_SAMPLES=raw_samples,
    )

    # Assertions
    assert acquisition.func == mock_func
    assert acquisition.gps == mock_gp_cons
    assert acquisition.cons == cons
    assert acquisition.device == device
    assert acquisition.q == q
    assert acquisition.NUM_RESTARTS == num_restarts
    assert acquisition.RAW_SAMPLES == raw_samples
    assert acquisition.nobj == mock_gp_cons.nobj
    assert acquisition.ncons == mock_gp_cons.ncons

def test_gp_posterior(mock_gp_no_cons, mock_func):
    # Mock GP models
    model_1 = Mock()
    model_1.posterior.return_value.sample.return_value = torch.tensor([[1.5], [1.6]])
    model_2 = Mock()
    model_2.posterior.return_value.sample.return_value = torch.tensor([[2.5], [2.6]])
    mock_gp_no_cons.models = [model_1, model_2]

    # Create Acquisition instance
    acquisition = Acquisition(func=mock_func, gps=mock_gp_no_cons)
    x = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    result = acquisition._gp_posterior(x, mock_gp_no_cons)

    # Assertions
    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([2, 2]), "Output shape is incorrect"
    model_1.posterior.assert_called_once()
    model_2.posterior.assert_called_once()

def test_gp_posterior_with_constraints():
    # Mock inputs
    func = Mock()
    func.dim = 2

    gps = Mock()
    gps.nobj = 2
    gps.ncons = 1  # Adding one constraint
    gps.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    
    # Objectives + constraints (nobj + ncons columns)
    gps.train_y = torch.tensor([
        [1.0, 2.0, -0.5],  # Infeasible point (constraint violated)
        [2.0, 1.0, 0.3]    # Feasible point
    ])

    # Mock GP models
    model_1 = Mock()
    model_1.posterior.return_value.sample.return_value = torch.tensor([[1.5, 2.6]])
    model_2 = Mock()
    model_2.posterior.return_value.sample.return_value = torch.tensor([[2.5, 3.6]])
    model_3 = Mock()
    model_3.posterior.return_value.sample.return_value = torch.tensor([[-1.0, 0.5]])
    gps.models = [model_1, model_2, model_3]

    # Create Acquisition instance
    acquisition = Acquisition(func=func, gps=gps)

    # Input tensor
    x = torch.tensor([[0.5, 0.5], [0.3, 0.7]])

    # Test gp_posterior
    result = acquisition._gp_posterior(x, gps)

    # Assertions
    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([2, 2]), f"Output shape is incorrect: {result.shape}"

    # Check that infeasible points are penalized
    assert torch.all(result[0, :] == 1e12), "Infeasible point not penalized"
    assert torch.any(result[1, :] != 1e12), "Feasible point incorrectly penalized"

    # Verify that posterior calls were made
    model_1.posterior.assert_called_once()
    model_2.posterior.assert_called_once()
    model_3.posterior.assert_called_once()

@patch("qpots.utils.pymoo_problem.nsga2")
@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_qpots(mock_func, real_func, q):
    pop_size = 100

    train_x = torch.rand(pop_size, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)
    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=pop_size, device=torch.device("cpu"))
    gps.fit_gp()
    acq = Acquisition(func=mock_func, gps=gps, q=q)

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    iteration = 1
    kwargs = {
        "nystrom": 0,
        "mt": 0,
        "partial_info": 0,
        "iters": 10,
        "dim": 2,
        "nychoice": "random",
        "q": q,
        "ngen": 10,
    }
    res = acq.qpots(bounds, iteration, **kwargs)

    assert isinstance(res, torch.Tensor), "Output is not a Tensor"
    assert res.shape == torch.Size([q, 2])

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_qpots_nystrom(real_func, q):
    pop_size = 100

    train_x = torch.rand(pop_size, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)
    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=pop_size, device=torch.device("cpu"))
    gps.fit_gp()
    acq = Acquisition(func=real_func, gps=gps, q=q)

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    iteration = 1
    kwargs = {
        "nystrom": 1,
        "mt": 0,
        "partial_info": 0,
        "iters": 10,
        "dim": 2,
        "nychoice": "random",
        "q": q,
        "ngen": 10,
    }
    res = acq.qpots(bounds, iteration, **kwargs)

    assert isinstance(res, torch.Tensor), "Output is not a Tensor"
    assert res.shape == torch.Size([q, 2])

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_qlogei(real_func, q):
    train_x = torch.rand(5, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    # Create GPs for each objective
    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=5, device=torch.device("cpu"))
    gps.fit_gp()

    # Create Acquisition instance
    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    # Reference point for hypervolume calculation
    ref_point = torch.tensor([-300.0, -18.0])

    # Call qlogei
    manual_seed(42)# Fix seed for reproducibility
    result = acquisition.qlogei(ref_point)

    # Assertions
    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_parego(real_func, q):
    train_x = torch.rand(5, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=5, device=torch.device("cpu"))
    gps.fit_gp()

    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    manual_seed(42)
    result = acquisition.parego()

    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_qlogparego(real_func, q):
    train_x = torch.rand(5, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=5, device=torch.device("cpu"))
    gps.fit_gp()

    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    manual_seed(42)
    result = acquisition.qlogparego()

    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_pesmo(real_func, q):
    train_x = torch.rand(100, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=100, device=torch.device("cpu"))
    gps.fit_gp()

    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    manual_seed(42)
    result = acquisition.pesmo()

    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"

@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_mesmo(real_func, q):
    train_x = torch.rand(100, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=100, device=torch.device("cpu"))
    gps.fit_gp()

    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    manual_seed(42)
    result = acquisition.mesmo()

    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"

@pytest.mark.parametrize("q", [1, 2, 3])
def test_jesmo(real_func, q):
    train_x = torch.rand(100, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(), nobj=real_func.nobj, ncons=0, ntrain=100, device=torch.device("cpu"))
    gps.fit_gp_no_variance()

    acquisition = Acquisition(func=real_func, gps=gps, q=q)

    manual_seed(42)
    result = acquisition.jesmo()

    assert isinstance(result, torch.Tensor), "Output is not a tensor"
    assert result.shape == torch.Size([q, 2]), f"Output shape is incorrect: {result.shape}"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Result is out of bounds"


# ---------------------------------------------------------------------------
# Tests added for improved coverage
# ---------------------------------------------------------------------------

def test_sobol(real_func):
    """sobol() returns a tensor within the unit hypercube for q=1."""
    train_x = torch.rand(10, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)
    gps = ModelObject(
        train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(),
        nobj=real_func.nobj, ncons=0, device=torch.device("cpu"),
    )
    gps.fit_gp()
    acq = Acquisition(func=real_func, gps=gps, q=1)

    result = acq.sobol()

    assert isinstance(result, torch.Tensor)
    assert result.shape[-1] == real_func.dim, "Last dimension must equal input dim"
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Sobol samples must be in [0, 1]"


def test_gp_posterior_sign_convention(mock_func):
    """_gp_posterior must negate GP samples so they are ready for NSGA-II minimization."""
    gps = Mock()
    gps.nobj = 2
    gps.ncons = 0
    gps.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    # Positive training data → mean >> 0 → unstandardised zero-sample = mean > 0
    gps.train_y = torch.tensor([[5.0, 10.0], [7.0, 12.0]], dtype=torch.float64)
    # Standardised sample of zero → unstandardise → mean of train_y (positive)
    model_1 = Mock()
    model_1.posterior.return_value.sample.return_value = torch.zeros(2, 1, dtype=torch.float64)
    model_2 = Mock()
    model_2.posterior.return_value.sample.return_value = torch.zeros(2, 1, dtype=torch.float64)
    gps.models = [model_1, model_2]

    acq = Acquisition(func=mock_func, gps=gps)
    x = torch.rand(2, 2, dtype=torch.float64)
    result = acq._gp_posterior(x, gps, seed_iter=1)

    assert (result < 0).all(), (
        "_gp_posterior must return -Ys so that NSGA-II can minimise (expected all negative)"
    )


def test_qpots_constrained(real_func):
    """qpots must handle ncons > 0 without error and return valid normalised candidates."""
    n = 50
    train_x = torch.rand(n, real_func.dim, dtype=torch.float64)
    train_y_obj = real_func.evaluate(train_x)
    # Synthetic feasible constraint column (all positive)
    train_y_con = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.cat([train_y_obj, train_y_con], dim=-1)

    gps = ModelObject(
        train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(),
        nobj=real_func.nobj, ncons=1, device=torch.device("cpu"),
    )
    gps.fit_gp()
    acq = Acquisition(func=real_func, gps=gps, q=2)

    kwargs = {
        "nystrom": 0, "iters": 5, "dim": real_func.dim,
        "nychoice": "random", "q": 2, "ngen": 5,
    }
    result = acq.qpots(real_func.get_bounds(), iteration=1, **kwargs)

    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([2, real_func.dim])
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Candidates must be in unit hypercube"


@pytest.mark.parametrize("q", [1, 2])
def test_qpots_output_in_unit_hypercube(real_func, q):
    """qpots normalises candidates; all returned values must lie in [0, 1]."""
    n = 50
    train_x = torch.rand(n, real_func.dim, dtype=torch.float64)
    train_y = real_func.evaluate(train_x)
    gps = ModelObject(
        train_x=train_x, train_y=train_y, bounds=real_func.get_bounds(),
        nobj=real_func.nobj, ncons=0, device=torch.device("cpu"),
    )
    gps.fit_gp()
    acq = Acquisition(func=real_func, gps=gps, q=q)

    kwargs = {
        "nystrom": 0, "iters": 5, "dim": real_func.dim,
        "nychoice": "random", "q": q, "ngen": 5,
    }
    result = acq.qpots(real_func.get_bounds(), iteration=1, **kwargs)

    assert (result >= -1e-6).all() and (result <= 1.0 + 1e-6).all(), (
        f"qpots candidates not in [0,1]^dim: min={result.min().item():.4f}, "
        f"max={result.max().item():.4f}"
    )
