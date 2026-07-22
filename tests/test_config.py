import pytest
import torch

from qpots.acquisition import Acquisition
from qpots.config import (
    RuntimeConfig,
    get_default_runtime,
    set_default_runtime,
)
from qpots.function import Function
from qpots.model_object import ModelObject


@pytest.fixture(autouse=True)
def restore_runtime():
    original = get_default_runtime()
    yield
    set_default_runtime(original)


def test_runtime_config_normalizes_device_string():
    runtime = RuntimeConfig(device="cpu", dtype=torch.float32)
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32


def test_set_default_runtime_changes_new_objects():
    set_default_runtime(device="cpu", dtype=torch.float32)
    function = Function(
        custom_func=lambda x: torch.column_stack((x[:, 0], x[:, 0])),
        bounds=torch.tensor([[0.0], [1.0]]),
        dim=1,
        nobj=2,
    )
    assert function.device == torch.device("cpu")
    assert function.dtype == torch.float32


def test_explicit_fields_override_runtime():
    runtime = RuntimeConfig(device="cpu", dtype=torch.float32)
    function = Function(
        custom_func=lambda x: torch.column_stack((x[:, 0], x[:, 0])),
        bounds=torch.tensor([[0.0], [1.0]]),
        dim=1,
        nobj=2,
        runtime=runtime,
        dtype=torch.float64,
    )
    assert function.dtype == torch.float64


def test_runtime_is_injected_into_model_and_acquisition():
    runtime = RuntimeConfig(device="cpu", dtype=torch.float32)
    bounds = torch.tensor([[0.0], [1.0]])
    function = Function(
        custom_func=lambda x: torch.column_stack((x[:, 0], x[:, 0])),
        bounds=bounds,
        dim=1,
        nobj=2,
        runtime=runtime,
    )
    model = ModelObject(
        torch.tensor([[0.5]]),
        torch.tensor([[1.0, 1.0]]),
        bounds,
        nobj=2,
        ncons=0,
        runtime=runtime,
    )
    acquisition = Acquisition(function, model, runtime=runtime)
    assert model.dtype == acquisition.dtype == torch.float32
    assert model.device == acquisition.device == torch.device("cpu")


@pytest.mark.parametrize("dtype", [torch.int64, "float64"])
def test_invalid_dtype_is_rejected(dtype):
    error = ValueError if isinstance(dtype, torch.dtype) else TypeError
    with pytest.raises(error):
        RuntimeConfig(dtype=dtype)


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="Invalid torch device"):
        RuntimeConfig(device="not-a-device")


def test_set_default_runtime_rejects_ambiguous_arguments():
    with pytest.raises(ValueError, match="either runtime or device/dtype"):
        set_default_runtime(RuntimeConfig(), device="cpu")
