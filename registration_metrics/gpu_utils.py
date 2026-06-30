"""Optional torch device helpers for GPU-accelerated metrics."""
from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("registration_metrics")

try:
    import torch
except ImportError:  # pragma: no cover - depends on environment
    torch = None  # type: ignore[assignment]


def get_device(use_gpu: bool, requested_device: str = "cuda:0") -> Any:
    """Return a torch.device-like object, falling back to CPU when torch/CUDA is unavailable."""
    if torch is None:
        LOGGER.warning("[GPU] use_gpu=%s requested_device=%s available=False final_device=cpu fallback_reason=torch not installed", use_gpu, requested_device)
        return "cpu"
    available = bool(use_gpu and torch.cuda.is_available())
    if use_gpu and not available:
        LOGGER.warning("[GPU] use_gpu=True requested_device=%s available=False final_device=cpu fallback_reason=CUDA not available", requested_device)
        return torch.device("cpu")
    final = requested_device if use_gpu else "cpu"
    LOGGER.info("[GPU] use_gpu=%s requested_device=%s available=%s final_device=%s", use_gpu, requested_device, available, final)
    return torch.device(final)


def device_name(device: Any) -> str:
    """Return a stable string name for a torch or fallback device."""
    return str(device)


def to_tensor_np(array, device: Any, dtype: Any = None):
    """Convert a numpy-like array to a torch tensor on the requested device."""
    if torch is None:
        raise RuntimeError("torch is not installed")
    dtype = dtype or torch.float32
    return torch.as_tensor(array, dtype=dtype, device=device)


def gpu_requested_for(metric_group: str, gpu_metrics: str | None) -> bool:
    """Return whether a metric group should attempt GPU based on a comma list or 'all'."""
    if not gpu_metrics:
        return False
    items = {x.strip().lower() for x in str(gpu_metrics).split(",") if x.strip()}
    return "all" in items or metric_group.lower() in items
