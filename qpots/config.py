"""
Central runtime configuration for qPOTS.

Edit ``DEFAULT_DTYPE`` to change the package-wide floating-point precision.
``DEFAULT_DEVICE`` automatically uses CUDA when a GPU is available and falls
back to CPU otherwise.
"""

from __future__ import annotations

from typing import Any

import torch


DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    """Return an explicit dtype or the configured qPOTS default dtype."""
    return dtype if isinstance(dtype, torch.dtype) else DEFAULT_DTYPE


def get_device(device: torch.device | str | None = None) -> torch.device:
    """Return an explicit device or the configured qPOTS default device."""
    if isinstance(device, (torch.device, str)):
        return torch.device(device)
    return DEFAULT_DEVICE


def tensor_kwargs(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.device | torch.dtype]:
    """Return keyword arguments for creating floating-point tensors."""
    return {"device": get_device(device), "dtype": get_dtype(dtype)}


def as_tensor(
    data: Any,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert data to a tensor on the configured qPOTS device and dtype."""
    return torch.as_tensor(data, **tensor_kwargs(device=device, dtype=dtype))


def to_runtime(
    tensor: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move a tensor to the configured qPOTS device and dtype."""
    return tensor.to(device=get_device(device), dtype=get_dtype(dtype))
