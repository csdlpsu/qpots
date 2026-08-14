"""Shared pytest configuration."""

import pytest
import torch

from qpots.config import get_default_runtime, set_default_runtime


@pytest.fixture(scope="session", autouse=True)
def force_cpu_runtime():
    """Keep the CI test contract independent of CUDA availability."""
    original = get_default_runtime()
    set_default_runtime(device="cpu", dtype=torch.float64)
    yield
    set_default_runtime(original)
