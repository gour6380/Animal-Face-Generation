"""Explicit FP32 device policy and synchronized, non-additive memory readings."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import psutil
import torch


def select_device(requested: str = "auto") -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable implicit MPS CPU fallback and restart the kernel before this benchmark.")
    enforce_fp32()
    available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
    if requested == "auto":
        return torch.device(next(name for name in ("mps", "cuda", "cpu") if available[name]))
    if requested not in available or not available[requested]:
        raise RuntimeError(f"Requested device is unavailable: {requested}")
    return torch.device(requested)


def enforce_fp32() -> None:
    """Disable NVIDIA tensor-float shortcuts; this experiment has no autocast."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: str | torch.device) -> dict[str, Any]:
    device = torch.device(device)
    result: dict[str, Any] = {"process_rss_bytes": psutil.Process().memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
        "meaning": "Instantaneous observations, not an exact stage peak; overlapping unified-memory counters must not be summed."}
    if device.type == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
    elif device.type == "cuda":
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device), cuda_reserved_bytes=torch.cuda.memory_reserved(device))
    return result
