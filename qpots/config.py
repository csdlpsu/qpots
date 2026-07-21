"""
Central runtime configuration for qPOTS.

Use :func:`set_default_runtime` to change package-wide precision or device
selection without modifying the installed source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class RuntimeConfig:
    """Device and floating-point precision used by qPOTS objects."""

    device: torch.device | str = DEFAULT_DEVICE
    dtype: torch.dtype = DEFAULT_DTYPE

    def __post_init__(self) -> None:
        try:
            resolved_device = torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid torch device: {self.device!r}") from exc
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be an instance of torch.dtype")
        if not self.dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        object.__setattr__(self, "device", resolved_device)


_DEFAULT_RUNTIME = RuntimeConfig()


def get_default_runtime() -> RuntimeConfig:
    """Return the default runtime configuration for newly created qPOTS objects."""
    return _DEFAULT_RUNTIME


def set_default_runtime(
    runtime: RuntimeConfig | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> RuntimeConfig:
    """Set and return the default runtime used by newly created qPOTS objects.

    A complete :class:`RuntimeConfig` may be supplied, or individual fields may
    be updated from the current default using keyword arguments.
    """
    global _DEFAULT_RUNTIME
    if runtime is not None and (device is not None or dtype is not None):
        raise ValueError("Pass either runtime or device/dtype overrides, not both")
    if runtime is not None and not isinstance(runtime, RuntimeConfig):
        raise TypeError("runtime must be a RuntimeConfig")
    current = get_default_runtime()
    _DEFAULT_RUNTIME = runtime or RuntimeConfig(
        device=current.device if device is None else device,
        dtype=current.dtype if dtype is None else dtype,
    )
    return _DEFAULT_RUNTIME


def resolve_runtime(
    runtime: RuntimeConfig | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> RuntimeConfig:
    """Resolve explicit fields over an object config and then package defaults."""
    if runtime is not None and not isinstance(runtime, RuntimeConfig):
        raise TypeError("runtime must be a RuntimeConfig")
    base = runtime or get_default_runtime()
    return RuntimeConfig(
        device=base.device if device is None else device,
        dtype=base.dtype if dtype is None else dtype,
    )


def get_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    """Return an explicit dtype or the configured qPOTS default dtype."""
    if dtype is None:
        return get_default_runtime().dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be an instance of torch.dtype")
    return RuntimeConfig(dtype=dtype).dtype


def get_device(device: torch.device | str | None = None) -> torch.device:
    """Return an explicit device or the configured qPOTS default device."""
    if device is None:
        return get_default_runtime().device
    return RuntimeConfig(device=device).device


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
