"""Offline two-process Gloo tests; no image data, pretrained networks or CUDA."""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from filelock import FileLock
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from src.checkpoints import atomic_torch_save, capture_rng, restore_rng
from src.config import ExperimentConfig, TrainingConfig
from src.distributed import (
    gather_rank_states,
    global_batches,
    rank_batches,
    restore_rank_state,
    run_distributed_epoch,
)
from src.io import atomic_json, read_json


def test_global_shards_are_sample_exact_and_use_a_private_rng():
    before = torch.get_rng_state().clone()
    for count in (1, 3, 11, 64, 12904):
        batches = list(global_batches(count, 64, seed=910))
        assert sorted(index for batch in batches for index in batch) == list(range(count))
        for index, global_batch in enumerate(batches):
            shards = [list(rank_batches(count, 64, 910, rank, 2))[index] for rank in range(2)]
            assert all(denominator == len(global_batch) for _, denominator in shards)
            assert sorted(shards[0][0] + shards[1][0]) == sorted(global_batch)
            assert not set(shards[0][0]) & set(shards[1][0])
    assert torch.equal(before, torch.get_rng_state())
    assert list(global_batches(19, 7, 10)) != list(global_batches(19, 7, 11))


def test_distributed_rng_requires_identical_topology_and_complete_rank_states():
    topology = {"world_size": 2, "backend": "gloo", "gpu_uuids": []}
    checkpoint = {"distributed": {"topology": topology, "rng_by_rank": [{"rank": 0}, {"rank": 1}]}}
    restored = []
    restore_rank_state(checkpoint, topology, 1, restored.append)
    assert restored == [{"rank": 1}]
    with pytest.raises(RuntimeError, match="topology"):
        restore_rank_state(checkpoint, {**topology, "world_size": 1}, 0, restored.append)
    with pytest.raises(RuntimeError, match="topology"):
        restore_rank_state(checkpoint, {**topology, "gpu_uuids": ["GPU-other"]}, 0, restored.append)
    checkpoint["distributed"]["rng_by_rank"].pop()
    with pytest.raises(RuntimeError, match="per-rank RNG"):
        restore_rank_state(checkpoint, topology, 0, restored.append)


def test_direct_training_cannot_bypass_saved_multi_gpu_assignments(monkeypatch):
    from src.train import _validate_training_assignment

    run = SimpleNamespace(hardware={"device": "cuda", "tokenizer_world_size": 2, "parallel_generators": True,
                                   "assignments": {"diffusion": ["GPU-0"], "autoregressive": ["GPU-1"]}})
    with pytest.raises(RuntimeError, match="two tokenizer workers"):
        _validate_training_assignment(run, "tokenizer")
    monkeypatch.delenv("AFG_WORKER_UUIDS", raising=False)
    monkeypatch.delenv("AFG_WORKER_ROLE", raising=False)
    with pytest.raises(RuntimeError, match="isolated GPU"):
        _validate_training_assignment(run, "diffusion")
    monkeypatch.setenv("AFG_WORKER_UUIDS", "GPU-0")
    monkeypatch.setenv("AFG_WORKER_ROLE", "diffusion")
    _validate_training_assignment(run, "diffusion")
    with pytest.raises(RuntimeError, match="isolated GPU"):
        _validate_training_assignment(run, "autoregressive")
    monkeypatch.setenv("AFG_WORKER_UUIDS", "GPU-1")
    monkeypatch.setenv("AFG_WORKER_ROLE", "autoregressive")
    _validate_training_assignment(run, "autoregressive")
    # Ordinary CPU/MPS and one-CUDA direct calls preserve the existing API.
    _validate_training_assignment(SimpleNamespace(hardware={"device": "mps"}), "tokenizer")
    _validate_training_assignment(SimpleNamespace(hardware={"device": "cuda", "parallel_generators": False}), "diffusion")


@pytest.mark.parametrize("backend", ["nccl", "gloo"])
@pytest.mark.parametrize("rank", [0, 1])
def test_tokenizer_process_group_uses_explicit_local_cuda_device(monkeypatch, capsys, backend, rank):
    """Validate CUDA arguments without initializing CUDA or a process group."""
    from src import train

    initialized = False
    calls = []
    run = SimpleNamespace(device="cuda" if backend == "nccl" else "cpu",
                          hardware={"assignments": {"tokenizer": ["GPU-first", "GPU-second"]}})

    def initialize(**kwargs):
        nonlocal initialized
        initialized = True
        calls.append(("initialize", kwargs))

    def destroy():
        nonlocal initialized
        initialized = False
        calls.append(("destroy", None))

    def train_group(received):
        assert received is run and initialized
        calls.append(("train", None))
        return {"status": "synthetic"}

    def cuda_count():
        assert backend == "nccl", "The Gloo path must not query CUDA."
        return 1

    def set_device(index):
        assert backend == "nccl", "The Gloo path must not select CUDA."
        calls.append(("device", index))

    monkeypatch.setenv("AFG_WORKER_UUIDS", run.hardware["assignments"]["tokenizer"][rank])
    monkeypatch.setattr(train.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(train.dist, "init_process_group", initialize)
    monkeypatch.setattr(train.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(train.torch.cuda, "device_count", cuda_count)
    monkeypatch.setattr(train.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(train, "_train_tokenizer_group", train_group)
    result = train.train_tokenizer_distributed(run, rank, 2, "file:///synthetic", backend)

    assert result == {"status": "synthetic"} and not initialized
    kwargs = next(value for name, value in calls if name == "initialize")
    assert kwargs["backend"] == backend and kwargs["rank"] == rank and kwargs["world_size"] == 2
    assert kwargs["init_method"] == "file:///synthetic"
    assert kwargs["timeout"].total_seconds() == 1800
    if backend == "nccl":
        assert kwargs["device_id"] == torch.device("cuda", 0)
        assert calls[0] == ("device", 0)
    else:
        assert "device_id" not in kwargs
        assert calls[0][0] == "initialize"
    assert [name for name, _ in calls][-2:] == ["train", "destroy"]
    output = capsys.readouterr().out
    assert f"[tokenizer rank {rank}/2] Initializing {backend}" in output
    assert "Process group initialized; preparing the run." in output


@pytest.mark.parametrize("backend,expected", [("nccl", {"device_ids": [0]}), ("gloo", {})])
def test_tokenizer_barrier_binds_only_nccl_device(monkeypatch, backend, expected):
    from src import train

    calls = []
    monkeypatch.setattr(train.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(train.dist, "barrier", lambda **kwargs: calls.append(kwargs))
    train._tokenizer_barrier()
    assert calls == [expected]


def test_tokenizer_rejects_wrong_uuid_before_initialization(monkeypatch):
    from src import train

    run = SimpleNamespace(device="cuda", hardware={"assignments": {"tokenizer": ["GPU-first", "GPU-second"]}})
    monkeypatch.setenv("AFG_WORKER_UUIDS", "GPU-first")
    monkeypatch.setattr(train.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(train.torch.cuda, "device_count", lambda: 1)

    def unexpected(*args, **kwargs):
        pytest.fail("A mismatched UUID must be rejected before selecting CUDA or creating a process group.")

    monkeypatch.setattr(train.torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(train.dist, "init_process_group", unexpected)
    with pytest.raises(RuntimeError, match="rank order"):
        train.train_tokenizer_distributed(run, 1, 2, "file:///synthetic")


def _data(count: int) -> list[dict[str, Any]]:
    return [{"x": torch.tensor([math.sin(index), index / 19, (index % 3) / 3]),
             "target": torch.tensor([math.cos(index)]), "label": index}
            for index in range(count)]


def _loss(model: nn.Module, batch: dict[str, Any]):
    loss = (model(batch["x"]) - batch["target"]).square().mean()
    return loss, {"loss": loss.detach().item()}


def _new_model(dropout: bool = False) -> nn.Module:
    torch.manual_seed(713)
    return nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.3) if dropout else nn.Tanh(), nn.Linear(4, 1))


def _worker(rank: int, rendezvous: str, directory: str) -> None:
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        config = replace(TrainingConfig(), microbatch=2, effective_batch=7, gradient_clip=1e6)
        # Includes unequal local microbatch counts, singleton global remainder,
        # and a rank receiving no real example in an update.
        for count in (1, 3, 10, 15, 19):
            data = _data(count)
            model, reference = _new_model(), _new_model()
            wrapped = DistributedDataParallel(model, broadcast_buffers=False)
            communications = []

            def hook(state, bucket, communications=communications):
                communications.append(1)
                return allreduce_hook(state, bucket)

            wrapped.register_comm_hook(None, hook)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.04)
            metrics, steps = run_distributed_epoch(
                wrapped, data, config, 2, lambda batch, wrapped=wrapped: _loss(wrapped, batch), optimizer, "synthetic")
            expected_loss = 0.0
            for indices in global_batches(count, config.effective_batch, config.seed + 10_002):
                batch = default_collate([data[index] for index in indices])
                reference_optimizer.zero_grad(set_to_none=True)
                loss, _ = _loss(reference, batch)
                loss.backward()
                reference_optimizer.step()
                expected_loss += loss.item() * len(indices)
            assert steps == math.ceil(count / config.effective_batch)
            assert len(communications) == steps
            assert metrics["loss"] == pytest.approx(expected_loss / count, rel=2e-6, abs=2e-7)
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        _check_interrupted_epoch(rank, directory, config)
        _check_training_integration(rank, directory, config)
    finally:
        dist.destroy_process_group()


def _check_interrupted_epoch(rank: int, directory: str, config: TrainingConfig) -> None:
    model = _new_model(dropout=True)
    wrapped = DistributedDataParallel(model, broadcast_buffers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    data = _data(19)
    torch.manual_seed(711 + rank)
    run_distributed_epoch(wrapped, data, config, 0, lambda batch: _loss(wrapped, batch), optimizer, "checkpoint")
    topology = {"world_size": 2, "backend": "gloo", "gpu_uuids": []}
    states = gather_rank_states(capture_rng("cpu"))
    checkpoint_path = Path(directory) / "synthetic_epoch.pt"
    if rank == 0:
        atomic_torch_save(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "epoch": 1, "distributed": {"topology": topology, "rng_by_rank": states}})
    dist.barrier()

    def restore():
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        assert not any(key.startswith("module.") for key in checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rank_state(checkpoint, topology, rank, restore_rng)

    restore()
    run_distributed_epoch(wrapped, data, config, 1, lambda batch: _loss(wrapped, batch), optimizer, "continuous")
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    expected_next_rng = torch.randn(7)
    restore()
    # This uncommitted partial epoch changes weights, Adam moments and RNG.
    run_distributed_epoch(wrapped, data[:7], config, 1, lambda batch: _loss(wrapped, batch), optimizer, "interrupted")
    restore()
    run_distributed_epoch(wrapped, data, config, 1, lambda batch: _loss(wrapped, batch), optimizer, "resumed")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.randn(7), expected_next_rng, rtol=0, atol=0)


def _check_training_integration(rank: int, directory: str, config: TrainingConfig) -> None:
    """Exercise the real epoch transaction path with tiny synthetic stand-ins."""
    from src import train

    path = Path(directory) / "synthetic_run"
    model_path = path / "models" / "tokenizer"
    manifest = {"digest": "synthetic-v1", "splits": {"train": _data(19), "validation": _data(3)}}
    if rank == 0:
        model_path.mkdir(parents=True)
        (path / "logs").mkdir()
        atomic_json(path / "run.json", {"source_sha256": "synthetic-source"})
        atomic_json(path / "data" / "manifest.json", {"digest": "synthetic-v1"})
    dist.barrier()
    experiment = ExperimentConfig()
    experiment = replace(experiment, tokenizer=replace(experiment.tokenizer, training=replace(config, epochs=2)))
    run = SimpleNamespace(root=Path(directory), path=path, device="cpu", config=experiment,
                          hardware={"assignments": {"tokenizer": []}}, verify=lambda: None,
                          model_dir=lambda family: model_path,
                          training_lock=lambda family: FileLock(path / ".tokenizer.lock", timeout=0))

    class TinyTokenizer(nn.Module):
        def __init__(self, configuration):
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, image):
            return {"reconstruction": self.linear(image),
                    "codebook_loss": self.linear.weight.square().mean(),
                    "commitment_loss": self.linear.bias.square().mean()}

    class TinyPerceptual(nn.Module):
        def forward(self, reconstruction, image):
            return (reconstruction - image).square().mean(dim=1)

    validations = []
    original_epoch = train._run_epoch

    def rank_zero_validation(*args, **kwargs):
        assert rank == 0, "Only rank zero may validate or select a checkpoint."
        validations.append(1)
        return original_epoch(*args, **kwargs)

    def images(root, records, image_size):
        return [{"image": value["x"], "label": value["label"]} for value in records]

    with patch.object(train, "VQAutoencoder", TinyTokenizer), \
         patch.object(train, "load_lpips", lambda run, device: TinyPerceptual()), \
         patch.object(train, "ImageDataset", images), \
         patch.object(train, "prepare_data", lambda run: manifest), \
         patch.object(train, "_run_epoch", rank_zero_validation):
        result = train.train_tokenizer(run)
        assert result["status"] == "complete" and result["completed_epochs"] == 2
        assert len(validations) == (2 if rank == 0 else 0)
        result = train.train_tokenizer(run)
        assert result["status"] == "reused"
    dist.barrier()
    state = torch.load(model_path / "final.pt", weights_only=True, map_location="cpu")
    assert state["epoch"] == 2 and state["global_step"] == 6
    assert len(state["distributed"]["rng_by_rank"]) == 2
    assert all(not key.startswith("module.") for key in state["model"])
    assert read_json(model_path / "status.json")["status"] == "complete"
    records = (model_path / "metrics.jsonl").read_text().splitlines()
    assert len(records) == 2


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="PyTorch needs Gloo for CPU DDP")
def test_two_rank_cpu_gradients_and_interrupted_recovery(tmp_path: Path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = mp.spawn(_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not processes.join(timeout=0.2):
            if time.monotonic() > deadline:
                pytest.fail("Two-process synthetic DDP did not finish within 90 seconds.")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
