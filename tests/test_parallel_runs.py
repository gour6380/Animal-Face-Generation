"""Synthetic lock and shared-input tests; no experiment assets are loaded."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import Timeout

from src.config import ExperimentConfig
from src.data import load_prepared_data
from src.io import atomic_json, digest_object
from src.runs import Run


def contexts(tmp_path: Path):
    for family in ("tokenizer", "diffusion", "autoregressive"):
        (tmp_path / "models" / family).mkdir(parents=True)
    return [Run(tmp_path, tmp_path, ExperimentConfig(), "cpu") for _ in range(3)]


def test_distinct_families_can_overlap_but_coordinator_and_duplicate_cannot(tmp_path):
    first, second, third = contexts(tmp_path)
    with first.training_lock("diffusion"), second.training_lock("autoregressive"):
        with pytest.raises(Timeout), third.lock():
            pass
        with pytest.raises(Timeout), third.training_lock("diffusion"):
            pass
    with third.lock(), pytest.raises(Timeout), first.training_lock("tokenizer"):
        pass
    with first.training_lock("tokenizer"):
        pass


def test_training_cannot_upgrade_to_shared_state_mutation(tmp_path):
    first, _, _ = contexts(tmp_path)
    with first.training_lock("diffusion"), pytest.raises(Timeout), first.lock():
        pass


def test_training_lock_releases_after_failure(tmp_path):
    first, second, _ = contexts(tmp_path)
    with pytest.raises(RuntimeError), first.training_lock("diffusion"):
        raise RuntimeError("synthetic worker failure")
    with second.training_lock("diffusion"):
        pass


def prepared_fixture(tmp_path):
    acquisition = {"dataset_root": "synthetic"}
    inventory = {"records": [], "acquisition_sha256": digest_object(acquisition)}
    inventory["digest"] = digest_object(inventory)
    manifest = {"inventory_sha256": inventory["digest"], "split_seed": 1001,
                "splits": {"train": [], "validation": [], "test": []}}
    manifest["digest"] = digest_object(manifest)
    for name, obj in (("acquisition", acquisition), ("inventory", inventory), ("manifest", manifest)):
        atomic_json(tmp_path / "data" / f"{name}.json", obj)
    return SimpleNamespace(path=tmp_path, config=ExperimentConfig())


def test_prepared_input_reader_never_writes_and_detects_corruption(tmp_path):
    run = prepared_fixture(tmp_path)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "data").iterdir()}
    assert load_prepared_data(run)["split_seed"] == 1001
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "data").iterdir()}
    path = tmp_path / "data" / "manifest.json"
    value = json.loads(path.read_text())
    value["split_seed"] = 1002
    atomic_json(path, value)
    with pytest.raises(ValueError, match="modified"):
        load_prepared_data(run)


def test_prepared_reader_rejects_acquisition_change_and_configuration_change(tmp_path):
    run = prepared_fixture(tmp_path)
    run.config = replace(run.config, data=replace(run.config.data, split_seed=12))
    with pytest.raises(ValueError, match="seed"):
        load_prepared_data(run)
    atomic_json(tmp_path / "data" / "acquisition.json", {"dataset_root": "changed"})
    with pytest.raises(ValueError, match="acquisition"):
        load_prepared_data(run)


def test_default_run_is_one_epoch_with_original_full_evaluation():
    config = ExperimentConfig()
    config.validate()
    for family in (config.tokenizer, config.diffusion, config.autoregressive):
        assert family.training.epochs == 1
        assert family.training.warmup_epochs == 0
        assert family.training.effective_batch == 64
    assert config.evaluation.ddim_steps == (25, 50, 100)
    assert config.evaluation.temperatures == (0.8, 1.0, 1.2)
    assert config.evaluation.benchmark_repetitions == 30
    assert config.evaluation.kid_subsets == 100


def test_benchmark_records_actual_gpu_and_rejects_the_other_device(monkeypatch):
    import torch

    from src.benchmark import benchmark_device_identity

    gpu = SimpleNamespace(uuid="GPU-first", name="synthetic GPU", major=12, minor=0, total_memory=24 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: gpu)
    run = SimpleNamespace(device="cuda", hardware={"selected_gpus": [{"uuid": "GPU-first"}]})
    assert benchmark_device_identity(run)["gpu"]["uuid"] == "GPU-first"
    gpu.uuid = "GPU-second"
    with pytest.raises(RuntimeError, match="first selected"):
        benchmark_device_identity(run)
