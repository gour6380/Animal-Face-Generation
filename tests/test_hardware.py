"""Synthetic topology and backend tests; these never execute GPU kernels."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from src.device import select_device
from src.hardware import (
    _check_cuda_architecture,
    build_hardware_plan,
    cuda_device_uuid,
    validate_hardware_plan,
    validate_worker_binding,
)


def inventory(count: int) -> list[dict]:
    return [{"index": index, "uuid": f"GPU-fake-{index}", "name": "Synthetic NVIDIA",
             "capability": [12, 0], "total_memory": 24 * 1024**3} for index in range(count)]


@pytest.mark.parametrize(("mps", "cuda", "expected"), [
    (True, True, "mps"), (True, False, "mps"), (False, True, "cuda"), (False, False, "cpu"),
])
def test_device_priority(monkeypatch: pytest.MonkeyPatch, mps: bool, cuda: bool, expected: str) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert select_device().type == expected
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_cuda_uses_first_two_gpus_and_fixed_fp32(count: int) -> None:
    plan = build_hardware_plan("cuda", inventory(count))
    selected = min(count, 2)
    assert len(plan["selected_gpus"]) == selected
    assert plan["tokenizer_world_size"] == selected
    assert plan["parallel_generators"] == (count >= 2)
    assert plan["assignments"]["autoregressive"] == [f"GPU-fake-{1 if count > 1 else 0}"]
    assert plan["dtype"] == "float32" and not plan["mixed_precision"]
    assert plan["unused_gpu_uuids"] == (["GPU-fake-2"] if count == 3 else [])


def test_mps_ignores_cuda_for_execution_but_records_inventory() -> None:
    plan = build_hardware_plan("mps", inventory(2))
    assert plan["selected_gpus"] == []
    assert plan["assignments"] == {"tokenizer": [], "diffusion": [], "autoregressive": []}
    assert not plan["parallel_generators"]


def test_uuid_normalizes_pytorch_bare_uuid() -> None:
    value = "12345678-1234-1234-1234-123456789abc"
    assert cuda_device_uuid(SimpleNamespace(uuid=value)) == "GPU-" + value
    with pytest.raises(RuntimeError, match="no usable UUID"):
        cuda_device_uuid(SimpleNamespace())


def test_corrupted_assignment_is_rejected() -> None:
    plan = build_hardware_plan("cuda", inventory(2))
    changed = copy.deepcopy(plan)
    changed["assignments"]["autoregressive"] = ["GPU-fake-0"]
    with pytest.raises(ValueError, match="assignments differ"):
        validate_hardware_plan(changed)


def test_blackwell_requires_compatible_build() -> None:
    _check_cuda_architecture([12, 0], ["sm_80", "sm_90", "sm_120"])
    _check_cuda_architecture([12, 0], ["sm_80", "compute_90"])
    with pytest.raises(RuntimeError, match="not supported"):
        _check_cuda_architecture([12, 0], ["sm_80", "sm_90"])


def test_worker_uuid_validated_without_trusting_cuda_ordinals(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_hardware_plan("cuda", inventory(2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid="GPU-fake-1"))
    monkeypatch.setenv("AFG_WORKER_UUIDS", "GPU-fake-1")
    monkeypatch.setenv("AFG_WORKER_ROLE", "autoregressive")
    validate_worker_binding(plan)
    monkeypatch.setenv("AFG_WORKER_ROLE", "diffusion")
    with pytest.raises(RuntimeError, match="assignment differs"):
        validate_worker_binding(plan)


def test_single_rank_may_see_one_of_tokenizer_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_hardware_plan("cuda", inventory(2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid="GPU-fake-1"))
    monkeypatch.setenv("AFG_WORKER_UUIDS", "GPU-fake-1")
    monkeypatch.setenv("AFG_WORKER_ROLE", "tokenizer")
    validate_worker_binding(plan)


def test_detect_hardware_reports_broken_cuda_instead_of_silent_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import hardware
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "nvidia_driver_inventory", lambda: {
        "available": True, "devices": [{"uuid": "GPU-fake-0", "driver": "580.0", "name": "GPU"}],
    })
    with pytest.raises(RuntimeError, match="will not silently run"):
        hardware.detect_hardware()
    assert hardware.detect_hardware("cpu")["device"] == "cpu"
