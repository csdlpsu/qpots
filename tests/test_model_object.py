import pytest
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from qpots.model_object import ModelObject

@pytest.fixture()
def mock_gp():
    train_x = torch.tensor([[0.4, 0.2]], dtype=torch.float64)
    train_y = torch.tensor([[40., 24.]], dtype=torch.float64)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    nobj = 2
    ncons = 2
    device = torch.device("cpu")
    noise_std = 1e-6

    gps = ModelObject(
        train_x=train_x, 
        train_y=train_y, 
        bounds=bounds, 
        nobj=nobj, 
        ncons=ncons, 
        device=device,
        noise_std=noise_std)
    
    return gps

def test_model_object_init():
    train_x = torch.tensor([[0.4, 0.2]])
    train_y = torch.tensor([[40., 24.]])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    nobj, ncons, noise_std = 2, 2, 1e-6
    device = torch.device("cpu")

    gps = ModelObject(train_x, train_y, bounds, nobj, ncons, device, noise_std)

    assert torch.equal(gps.train_x, train_x)
    assert torch.equal(gps.train_y, train_y)
    assert torch.equal(gps.bounds, bounds)
    assert gps.nobj == nobj
    assert gps.ncons == ncons
    assert gps.device == device
    assert gps.noise_std == noise_std

def test_fit_gp(mock_gp):
    mock_gp.fit_gp()
    assert len(mock_gp.models) == mock_gp.nobj
    assert len(mock_gp.mlls) == mock_gp.nobj
    assert all(isinstance(model, SingleTaskGP) for model in mock_gp.models)
    assert all(isinstance(mll, ExactMarginalLogLikelihood) for mll in mock_gp.mlls)

def test_fit_gp_no_variance(mock_gp):
    mock_gp.fit_gp_no_variance()
    assert len(mock_gp.models) == mock_gp.nobj
    assert len(mock_gp.mlls) == mock_gp.nobj
    assert all(isinstance(model, SingleTaskGP) for model in mock_gp.models)
    assert all(isinstance(mll, ExactMarginalLogLikelihood) for mll in mock_gp.mlls)

def test_model_predictions(mock_gp):
    mock_gp.fit_gp()
    test_x = torch.tensor([[0.5, 0.3]], dtype=torch.float64)
    preds = mock_gp.models[0].posterior(test_x)
    assert preds.mean.shape == (1, 1)

def test_no_objectives():
    with pytest.raises(IndexError):
        gps = ModelObject(torch.rand(1, 2), torch.rand(1, 0), torch.rand(2, 2), nobj=0, ncons=0, device="cpu")
        gps.fit_gp()



