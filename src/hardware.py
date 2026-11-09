"""Explicit, model-free hardware discovery and immutable worker assignments."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any
from uuid import UUID


def cuda_device_uuid(properties: Any) -> str:
    """Use the visible device's identity, never a remapped physical ordinal."""
    value = str(getattr(properties, "uuid", "")).strip()
    if value.startswith(("GPU-", "MIG-")) and len(value) > 4:
        return value
    try:
        identifier = UUID(value)
        if identifier.int:
            return f"GPU-{identifier}"
    except (ValueError, AttributeError):
        pass
    raise RuntimeError(f"CUDA device has no usable UUID: {value!r}. Update the CUDA PyTorch build.")


def build_hardware_plan(device: str, gpus: list[dict[str, Any]], *, host: dict[str, Any] | None = None,
                        torch_build: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure planning helper: select at most two visible GPUs without changing batches."""
    if device not in {"mps", "cuda", "cpu"} or (device == "cuda" and not gpus):
        raise ValueError("A CUDA plan requires at least one visible GPU.")
    selected = gpus[:2] if device == "cuda" else []
    uuids = [gpu["uuid"] for gpu in selected]
    parallel = len(selected) == 2
    plan = {"version": 1, "device": device,
            "strategy": "cuda_multiple" if parallel else "cuda_single" if device == "cuda" else device,
            "available_gpus": gpus, "selected_gpus": selected,
            "unused_gpu_uuids": [gpu["uuid"] for gpu in gpus[len(selected):]],
            "tokenizer_world_size": 2 if parallel else 1, "parallel_generators": parallel,
            "assignments": {"tokenizer": uuids, "diffusion": uuids[:1],
                            "autoregressive": uuids[1:2] if parallel else uuids[:1]},
            "dtype": "float32", "mixed_precision": False, "tf32": False,
            "host": host or {}, "torch_build": torch_build or {}}
    validate_hardware_plan(plan)
    return plan


def validate_hardware_plan(plan: dict[str, Any]) -> None:
    """Reject changed or contradictory saved assignments without probing hardware."""
    if plan.get("version") != 1 or plan.get("device") not in {"mps", "cuda", "cpu"}:
        raise ValueError("Unsupported hardware plan.")
    gpus = plan.get("available_gpus")
    if not isinstance(gpus, list):
        raise TypeError("Hardware plan requires its complete visible GPU inventory.")
    seen = set()
    for index, gpu in enumerate(gpus):
        if gpu.get("index") != index or not isinstance(gpu.get("uuid"), str) or not gpu["uuid"].startswith(("GPU-", "MIG-")):
            raise ValueError("GPU inventory indices and UUIDs are invalid.")
        if gpu["uuid"] in seen:
            raise ValueError("Visible CUDA GPU UUIDs must be distinct.")
        seen.add(gpu["uuid"])
        if not isinstance(gpu.get("total_memory"), int) or gpu["total_memory"] <= 0:
            raise ValueError("GPU memory capacity must be recorded.")
        if not isinstance(gpu.get("name"), str) or len(gpu.get("capability", [])) != 2:
            raise ValueError("GPU name and compute capability must be recorded.")
    selected = gpus[:2] if plan["device"] == "cuda" else []
    if plan["device"] == "cuda" and not selected:
        raise ValueError("No GPUs selected for CUDA.")
    uuids = [gpu["uuid"] for gpu in selected]
    parallel = len(selected) == 2
    expected = {"tokenizer": uuids, "diffusion": uuids[:1],
                "autoregressive": uuids[1:2] if parallel else uuids[:1]}
    strategy = "cuda_multiple" if parallel else "cuda_single" if plan["device"] == "cuda" else plan["device"]
    if (plan.get("selected_gpus") != selected or plan.get("assignments") != expected
            or plan.get("tokenizer_world_size") != (2 if parallel else 1)
            or plan.get("parallel_generators") is not parallel or plan.get("strategy") != strategy
            or plan.get("unused_gpu_uuids") != [gpu["uuid"] for gpu in gpus[len(selected):]]):
        raise ValueError("Hardware assignments differ from the first-two-GPU policy.")
    if plan.get("dtype") != "float32" or plan.get("mixed_precision") is not False or plan.get("tf32") is not False:
        raise ValueError("This experiment requires FP32 with AMP and TF32 disabled.")


def _check_cuda_architecture(capability: list[int], architectures: list[str]) -> None:
    """Reject known incompatible builds before the first training allocation."""
    sass, ptx = [], []
    for architecture in architectures:
        match = re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", architecture)
        if match:
            (sass if match[1] == "sm" else ptx).append(int(match[2]))
    target = capability[0] * 10 + capability[1]
    if sass or ptx:
        compatible = any(code // 10 == target // 10 and code <= target for code in sass)
        compatible |= any(code <= target for code in ptx)
        if not compatible:
            raise RuntimeError(
                f"CUDA capability {capability} is not supported by this PyTorch build ({architectures}). "
                "Install the project's CUDA requirements in the notebook interpreter before training."
            )


def nvidia_driver_inventory() -> dict[str, Any]:
    """Read optional driver information without requiring NVIDIA tooling on macOS."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version,name",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=5, check=True).stdout
        devices = []
        for line in output.splitlines():
            uuid, driver, name = (part.strip() for part in line.split(",", maxsplit=2))
            devices.append({"uuid": uuid, "driver": driver, "name": name})
        return {"available": True, "devices": devices}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "reason": type(error).__name__, "devices": []}


def detect_hardware(requested: str = "auto") -> dict[str, Any]:
    """Discover MPS first, then CUDA, then CPU. No models or downloads are involved."""
    import psutil
    import torch

    from .device import select_device
    drivers = nvidia_driver_inventory()
    if (requested == "auto" and not torch.backends.mps.is_available()
            and drivers["devices"] and not torch.cuda.is_available()):
        raise RuntimeError("NVIDIA GPUs are visible to nvidia-smi but PyTorch CUDA is unavailable. "
                           "Check the container GPU allocation, driver, and CUDA PyTorch installation; "
                           "this notebook will not silently run the cloud test on CPU.")
    device = select_device(requested).type
    if device == "cuda" and torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 13:
        versions = [row["driver"] for row in drivers["devices"]]
        if any(int(value.split(".")[0]) < 580 for value in versions if value.split(".")[0].isdigit()):
            raise RuntimeError("The installed CUDA 13 PyTorch build requires NVIDIA driver 580 or later. "
                               "Select a compatible RunPod image/host before training.")
    gpus = []
    architectures = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = [properties.major, properties.minor]
            if device == "cuda" and index < 2:
                _check_cuda_architecture(capability, architectures)
            gpus.append({"index": index, "uuid": cuda_device_uuid(properties), "name": properties.name,
                         "capability": capability, "total_memory": properties.total_memory})
    return build_hardware_plan(device, gpus,
        host={"system": platform.system(), "machine": platform.machine(),
              "cpu_count": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total},
        torch_build={"version": str(torch.__version__), "cuda": torch.version.cuda,
                     "architectures": architectures, "nvidia_driver": drivers})


def validate_worker_binding(plan: dict[str, Any]) -> None:
    """Check masked workers against the full saved topology, independent of ordinals."""
    import torch

    from .device import select_device
    validate_hardware_plan(plan)
    if select_device(plan["device"]).type != plan["device"]:
        raise RuntimeError("Worker backend differs from the saved hardware plan.")
    if plan["device"] != "cuda":
        return
    actual = [cuda_device_uuid(torch.cuda.get_device_properties(index))
              for index in range(torch.cuda.device_count())]
    mask = os.environ.get("AFG_WORKER_UUIDS")
    if mask is None:
        expected = [gpu["uuid"] for gpu in plan["available_gpus"]]
    else:
        expected = mask.split(",") if mask else []
        if not expected or len(set(expected)) != len(expected):
            raise RuntimeError("CUDA worker has an empty or duplicate GPU assignment.")
        role = os.environ.get("AFG_WORKER_ROLE", "coordinator")
        allowed = plan["assignments"].get(role, plan["assignments"]["diffusion"])
        if role == "tokenizer":
            if expected != allowed and not (len(expected) == 1 and expected[0] in allowed):
                raise RuntimeError("Tokenizer worker assignment differs from its saved topology.")
        elif expected != allowed:
            raise RuntimeError(f"{role} GPU assignment differs from its saved topology.")
    if actual != expected:
        raise RuntimeError(f"Visible CUDA UUIDs {actual} differ from worker assignment {expected}.")
