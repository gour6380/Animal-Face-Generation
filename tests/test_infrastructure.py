"""Infrastructure regression tests, authored for the user and never run at handoff."""
from __future__ import annotations

import json
import random
import shutil
import stat
import sys
import zipfile
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from filelock import Timeout
from torch import nn

from src import checkpoints, data, device, runs, weights
from src.config import (
    AutoregressiveConfig,
    ExperimentConfig,
    TrainingConfig,
    config_from_dict,
)
from src.io import atomic_json, digest_object, read_json, sha256_file


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "config.py").write_text("# synthetic source fixture\n")
    (root / "requirements.txt").write_text("# synthetic dependency fixture\n")
    (root / ".python-version").write_text("3.13\n")
    (root / "notebooks").mkdir()
    atomic_json(root / "notebooks" / "experiment.ipynb", {
        "cells": [{"cell_type": "code", "source": ["RUN_NAME = None"],
                   "outputs": [], "execution_count": None, "metadata": {}}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    })
    monkeypatch.setattr(runs, "select_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(runs, "environment_identity", lambda selected: {
        "device": selected, "python": "synthetic-py313", "packages": {},
    })
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    return root


@pytest.fixture
def run(project: Path) -> runs.Run:
    value = runs.create_run(project, ExperimentConfig())
    atomic_json(value.path / "data" / "manifest.json", {"synthetic": True})
    return value


def test_configuration_json_round_trip_preserves_tuple_fields() -> None:
    config = ExperimentConfig()
    restored = config_from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.diffusion.channels, tuple)
    assert isinstance(restored.tokenizer.training.betas, tuple)
    assert isinstance(restored.evaluation.benchmark_batches, tuple)


def test_configuration_rejects_unequal_training_exposure() -> None:
    config = ExperimentConfig()
    changed = replace(config, autoregressive=replace(config.autoregressive,
        training=replace(config.autoregressive.training, epochs=201)))
    with pytest.raises(ValueError, match="equal epochs"):
        changed.validate()


def test_fresh_runs_increment_and_deleted_number_is_not_reused(project: Path) -> None:
    first = runs.create_run(project, ExperimentConfig())
    second = runs.create_run(project, ExperimentConfig())
    assert first.path.name == "run_001" and second.path.name == "run_002"
    shutil.rmtree(second.path)
    third = runs.create_run(project, ExperimentConfig())
    assert third.path.name == "run_003"
    reopened = runs.open_run(project, first.path.name)
    assert reopened.path == first.path
    assert reopened.hardware == first.hardware


def test_saved_hardware_plan_cannot_change_unnoticed(run: runs.Run) -> None:
    path = run.path / "run.json"
    metadata = read_json(path)
    metadata["hardware"]["tokenizer_world_size"] = 2
    atomic_json(path, metadata)
    with pytest.raises(ValueError, match="hardware plan is corrupted"):
        run.verify()


def test_independent_writers_cannot_lock_the_same_run(run: runs.Run) -> None:
    second_writer = runs.Run(run.root, run.path, run.config, run.device)
    with run.lock():
        # Calls on the same run context are deliberately reentrant.
        with run.lock():
            pass
        with pytest.raises(Timeout), second_writer.lock():
            raise AssertionError("A competing writer acquired the run lock.")
    # Leaving the context releases the OS lock for a separate writer.
    with second_writer.lock():
        pass


def test_source_change_rejects_resume(run: runs.Run) -> None:
    (run.root / "src" / "config.py").write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="[Ss]ource"):
        runs.open_run(run.root, run.path.name)


def test_saved_source_snapshot_cannot_change_unnoticed(run: runs.Run) -> None:
    (run.path / "source_snapshot" / "src" / "config.py").write_text("# corrupted snapshot\n")
    with pytest.raises(ValueError, match="[Ss]napshot"):
        run.verify()


def test_saved_configuration_cannot_change_unnoticed(run: runs.Run) -> None:
    path = run.path / "config" / "experiment.json"
    value = read_json(path)
    value["tokenizer"]["training"]["epochs"] += 1
    atomic_json(path, value)
    with pytest.raises(ValueError, match="[Cc]onfig"):
        run.verify()


def test_notebook_outputs_do_not_invalidate_resume(run: runs.Run) -> None:
    path = run.root / "notebooks" / "experiment.ipynb"
    notebook = read_json(path)
    notebook["cells"][0].update(execution_count=12, outputs=[{
        "output_type": "stream", "name": "stdout", "text": "created run_001\n",
    }])
    notebook["cells"][0]["metadata"]["execution"] = {"time": "synthetic"}
    atomic_json(path, notebook)
    run.verify()


def test_editing_notebook_run_selector_allows_explicit_resume(run: runs.Run) -> None:
    path = run.root / "notebooks" / "experiment.ipynb"
    notebook = read_json(path)
    notebook["cells"][0]["source"] = ['RUN_NAME = "run_001"']
    atomic_json(path, notebook)
    assert runs.open_run(run.root, "run_001").path == run.path
    saved = read_json(run.path / "source_snapshot" / "notebooks" / "experiment.ipynb")
    assert saved["cells"][0]["source"] == ["RUN_NAME = None"]


def test_saved_notebook_snapshot_still_has_integrity_protection(run: runs.Run) -> None:
    path = run.path / "source_snapshot" / "notebooks" / "experiment.ipynb"
    path.write_text("corrupted historical notebook")
    with pytest.raises(ValueError, match="[Ss]napshot"):
        run.verify()


def test_explicit_resume_rejects_different_resolved_configuration(run: runs.Run) -> None:
    changed = replace(run.config, tokenizer=replace(run.config.tokenizer, commitment_weight=0.5))
    with pytest.raises(ValueError, match="[Cc]onfig"):
        runs.open_run(run.root, run.path.name, config=changed)


def test_saved_environment_record_cannot_change_unnoticed(run: runs.Run) -> None:
    path = run.path / "environment" / "environment.json"
    value = read_json(path)
    value["python"] = "modified"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="[Ee]nvironment"):
        run.verify()


def test_notebook_analysis_override_does_not_rewrite_original_run_configuration(run: runs.Run) -> None:
    changed = replace(run.config, evaluation=replace(run.config.evaluation, sampling_seed=7001))
    original_hash = sha256_file(run.path / "config" / "experiment.json")
    reopened = runs.open_run(run.root, run.path.name, config=changed)
    assert reopened.config.evaluation.sampling_seed == run.config.evaluation.sampling_seed
    assert reopened.config.evaluation.sampling_seed != changed.evaluation.sampling_seed
    assert sha256_file(run.path / "config" / "experiment.json") == original_hash


@pytest.mark.parametrize("member_name", ["../escape.txt", "/absolute/escape.txt", "train/../../escape.txt"])
def test_archive_rejects_escaping_members(tmp_path: Path, member_name: str) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("safe.txt", "must not extract before full validation")
        source.writestr(member_name, "bad")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        data.safe_extract(archive, destination)
    assert not (destination / "safe.txt").exists()


def test_archive_rejects_symlink_members(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(link, "../../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        data.safe_extract(archive, tmp_path / "extracted")


def test_archive_normal_tree_extracts_exact_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("afhq/train/cat/a.png", b"synthetic bytes")
    destination = tmp_path / "extracted"
    data.safe_extract(archive, destination)
    assert (destination / "afhq/train/cat/a.png").read_bytes() == b"synthetic bytes"


def inventory_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict]:
    root = tmp_path / "dataset_project"
    run_path = root / "runs" / "run_001"
    run_path.mkdir(parents=True)
    config = ExperimentConfig()
    records = []
    for label, name in enumerate(config.data.classes):
        for partition, count in (("train", 10), ("val", 4)):
            for index in range(count):
                identity = f"{partition}/{name}/{index:03d}.png"
                relative = f"data/raw/afhq_v2/{identity}"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(identity.encode())
                digest = sha256_file(path)
                records.append({"id": identity, "relative_path": relative,
                    "label": label, "class_name": name, "source_split": partition,
                    "file_sha256": digest, "pixel_sha256": digest})
    config = replace(config, data=replace(config.data, expected_total=len(records)))
    fake = SimpleNamespace(root=root, path=run_path, config=config,
                           lock=lambda: nullcontext(), verify=lambda: None)
    acquisition = {"dataset_root": "data/raw/afhq_v2"}
    inventory = {"acquisition_sha256": digest_object(acquisition), "records": records, "count": len(records)}
    inventory["digest"] = digest_object(inventory)
    atomic_json(run_path / "data" / "inventory.json", inventory)
    atomic_json(run_path / "data" / "acquisition.json", acquisition)
    return fake, inventory


def test_afhq_upstream_validation_is_only_final_test(tmp_path: Path) -> None:
    fake, _ = inventory_fixture(tmp_path)
    first = data.prepare_data(fake)
    second = data.prepare_data(fake)
    assert first == second
    splits = first["splits"]
    assert all(row["source_split"] == "val" for row in splits["test"])
    assert all(row["source_split"] == "train" for name in ("train", "validation") for row in splits[name])
    identities = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    assert not identities["train"] & identities["validation"]
    assert not identities["train"] & identities["test"]
    assert not identities["validation"] & identities["test"]
    assert all(sum(row["label"] == label for row in splits["validation"]) == 1 for label in range(3))
    assert first["n_per_class"] == 4


def test_exact_pixel_duplicates_cannot_cross_development_splits(tmp_path: Path) -> None:
    fake, inventory = inventory_fixture(tmp_path)
    # All same-class development images collide, so any train/validation split leaks.
    for row in inventory["records"]:
        if row["label"] == 0 and row["source_split"] == "train":
            row["pixel_sha256"] = "same-pixel-content"
    inventory["digest"] = digest_object({key: value for key, value in inventory.items() if key != "digest"})
    atomic_json(fake.path / "data" / "inventory.json", inventory)
    with pytest.raises(ValueError, match="duplicated pixels"):
        data.prepare_data(fake)


def test_changed_dataset_file_is_rejected_before_split_creation(tmp_path: Path) -> None:
    fake, inventory = inventory_fixture(tmp_path)
    (fake.root / inventory["records"][0]["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset file changed"):
        data.prepare_data(fake)


def epoch_payload(epoch: int) -> dict:
    return {"epoch": epoch, "global_step": epoch * 10,
            "model": {"weight": torch.tensor([float(epoch)])}, "metadata": {}}


def test_epoch_commit_reconciles_uncommitted_metric_rows(run: runs.Run) -> None:
    checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=False)
    directory = run.model_dir("diffusion")
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"epoch": 2, "loss": 0.5}) + "\n")
    recovered = checkpoints.load_training_state(run, "diffusion")
    assert recovered["epoch"] == 1
    assert [json.loads(line)["epoch"] for line in (directory / "metrics.jsonl").read_text().splitlines()] == [1]
    assert len((directory / "metrics.csv").read_text().splitlines()) == 2


def test_committed_epoch_recovers_interrupted_alias_publication(run: runs.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    materialize = checkpoints._materialize
    def interrupted(*args: object) -> None:
        raise OSError("synthetic interruption after pointer commit")
    monkeypatch.setattr(checkpoints, "_materialize", interrupted)
    with pytest.raises(OSError, match="synthetic"):
        checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=False)
    monkeypatch.setattr(checkpoints, "_materialize", materialize)
    recovered = checkpoints.load_training_state(run, "diffusion")
    assert recovered["epoch"] == 1
    assert (run.model_dir("diffusion") / "last.pt").is_file()
    assert (run.model_dir("diffusion") / "best_validation.pt").is_file()


def test_corrupted_canonical_checkpoint_is_never_silently_used(run: runs.Run) -> None:
    checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=False)
    directory = run.model_dir("diffusion")
    pointer = read_json(directory / "commit.json")
    (directory / pointer["last"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        checkpoints.load_training_state(run, "diffusion")


def test_inference_verifies_only_the_selected_checkpoint(run: runs.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=False)
    checkpoints.save_epoch(run, "diffusion", epoch_payload(2), {"loss": 2.0}, is_best=False, final=True)
    directory = run.model_dir("diffusion")
    pointer = read_json(directory / "commit.json")
    reads: list[Path] = []
    original = checkpoints.sha256_file

    def observed(path: Path) -> str:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(checkpoints, "sha256_file", observed)
    state = checkpoints.load_model_checkpoint(run, "diffusion", "final")
    assert state["epoch"] == 2
    assert directory / pointer["final"]["path"] in reads
    assert directory / "final.pt" in reads
    assert directory / pointer["best_validation"]["path"] not in reads
    assert directory / "best_validation.pt" not in reads
    assert directory / "last.pt" not in reads


def test_materialization_hashes_shared_epoch_only_once(run: runs.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=True)
    directory = run.model_dir("diffusion")
    pointer = read_json(directory / "commit.json")
    reads: list[Path] = []
    original = checkpoints.sha256_file

    def observed(path: Path) -> str:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(checkpoints, "sha256_file", observed)
    checkpoints._materialize(directory, pointer)
    assert reads.count(directory / pointer["last"]["path"]) == 1


def test_completed_fit_cannot_accept_more_epoch_commits(run: runs.Run) -> None:
    checkpoints.save_epoch(run, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=True)
    with pytest.raises(ValueError, match="[Cc]omplete|[Ff]inal"):
        checkpoints.save_epoch(run, "diffusion", epoch_payload(2), {"loss": 0.5}, is_best=True, final=False)


def test_completed_stage_reuses_verified_checkpoint_without_training(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import train

    config = ExperimentConfig()
    config = replace(config,
        diffusion=replace(config.diffusion, training=replace(config.diffusion.training, epochs=1, warmup_epochs=0)),
        autoregressive=replace(config.autoregressive, training=replace(config.autoregressive.training, epochs=1, warmup_epochs=0)))
    completed = runs.create_run(project, config)
    atomic_json(completed.path / "data" / "manifest.json", {"synthetic": True})
    checkpoints.save_epoch(completed, "diffusion", epoch_payload(1), {"loss": 1.0}, is_best=True, final=True)
    final_path = completed.model_dir("diffusion") / "final.pt"
    original_hash = sha256_file(final_path)

    def forbid_training_work(*args: object, **kwargs: object) -> None:
        raise AssertionError("Completed-stage reuse must not prepare data or construct/train a model.")

    monkeypatch.setattr(train, "prepare_data", forbid_training_work)
    monkeypatch.setattr(train, "DiffusionUNet", forbid_training_work)
    monkeypatch.setattr(train, "_run_epoch", forbid_training_work)
    result = train.train_diffusion(completed)
    assert result["status"] == "reused" and result["completed_epochs"] == 1
    assert sha256_file(final_path) == original_hash


def test_architecture_json_and_state_dict_reconstruct_the_same_model(project: Path) -> None:
    from src.models.autoregressive import ARTransformer

    config = ExperimentConfig()
    config = replace(config, autoregressive=replace(config.autoregressive,
        layers=1, width=16, heads=2, feedforward_dim=32, dropout=0.0))
    recorded = runs.create_run(project, config)
    atomic_json(recorded.path / "data" / "manifest.json", {"synthetic": True})
    original = ARTransformer(config.autoregressive,
        codebook_size=config.tokenizer.codebook_size,
        sequence_length=config.latent_size ** 2,
        num_classes=len(config.data.classes)).eval()
    checkpoints.save_architecture(recorded, "autoregressive", original)
    payload = {**epoch_payload(1), "model": original.state_dict()}
    checkpoints.save_epoch(recorded, "autoregressive", payload, {"loss": 1.0}, is_best=True, final=False)

    directory = recorded.model_dir("autoregressive")
    description = read_json(directory / "architecture.json")
    model_settings = dict(description["model_config"])
    training_settings = dict(model_settings.pop("training"))
    training_settings["betas"] = tuple(training_settings["betas"])
    restored_config = AutoregressiveConfig(**model_settings, training=TrainingConfig(**training_settings))
    restored = ARTransformer(restored_config,
        codebook_size=description["codebook_size"],
        sequence_length=description["latent_size"] ** 2,
        num_classes=len(description["classes"])).eval()
    checkpoint = checkpoints.load_model_checkpoint(recorded, "autoregressive", "last")
    restored.load_state_dict(checkpoint["model"], strict=True)
    inputs = torch.tensor([[config.tokenizer.codebook_size, 1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)
    assert restored_config == config.autoregressive
    assert (directory / "architecture.txt").read_text()


def test_interrupted_epoch_resume_restores_python_numpy_and_torch_rng(
    run: runs.Run, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_state = checkpoints.capture_rng()
    try:
        random.seed(31)
        np.random.seed(37)
        torch.manual_seed(41)
        payload = {**epoch_payload(1), "rng": checkpoints.capture_rng()}
        checkpoints.save_epoch(run, "diffusion", payload, {"loss": 1.0}, is_best=True, final=False)
        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)
        # An interrupted, uncommitted epoch advances all streams arbitrarily.
        random.seed(91)
        np.random.seed(97)
        torch.manual_seed(101)
        random.random(), np.random.random(9), torch.rand(12)
        state = checkpoints.load_training_state(run, "diffusion")
        checkpoints.restore_rng(state["rng"])
        assert [random.random() for _ in range(4)] == expected_python
        assert np.array_equal(np.random.random(4), expected_numpy)
        assert torch.equal(torch.rand(4), expected_torch)
    finally:
        checkpoints.restore_rng(original_state)


def test_no_checkpoint_clears_all_uncommitted_metric_views(run: runs.Run) -> None:
    directory = run.model_dir("diffusion")
    (directory / "metrics.jsonl").write_text('{"epoch": 1, "loss": 2.0}\n')
    (directory / "metrics.csv").write_text("epoch,loss\n1,2.0\n")
    assert checkpoints.load_training_state(run, "diffusion") is None
    assert not (directory / "metrics.jsonl").read_text()
    assert not (directory / "metrics.csv").read_text()


def test_device_preference_and_explicit_unavailable_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.select_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device.select_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.select_device().type == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        device.select_device("mps")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="fallback"):
        device.select_device()


class FakePerceptualNetwork(nn.Module):
    """Mimic pinned LPIPS registration names without a large VGG allocation."""
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        assert kwargs["pretrained"] is False and kwargs["pnet_rand"] is True
        self.net = nn.Module()
        self.net.slice1 = nn.Sequential(OrderedDict([("0", nn.Conv2d(3, 2, 1))]))
        self.net.slice2 = nn.Sequential(OrderedDict([("4", nn.Conv2d(2, 2, 1))]))
        linear_layers = []
        for index in range(5):
            layer = nn.Module()
            layer.model = nn.Sequential(nn.Dropout(), nn.Conv2d(2, 1, 1, bias=False))
            setattr(self, f"lin{index}", layer)
            linear_layers.append(layer)
        self.lins = nn.ModuleList(linear_layers)


def test_offline_lpips_loader_maps_original_vgg_indices_and_calibration_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=FakePerceptualNetwork))
    backbone = {
        "features.0.weight": torch.full((2, 3, 1, 1), 0.25),
        "features.0.bias": torch.full((2,), 0.5),
        "features.4.weight": torch.full((2, 2, 1, 1), 0.75),
        "features.4.bias": torch.full((2,), 1.0),
    }
    calibration = {f"lin{index}.model.1.weight": torch.full((1, 2, 1, 1), float(index + 1))
                   for index in range(5)}
    paths = {"vgg16": tmp_path / "backbone.pt", "lpips": tmp_path / "calibration.pt"}
    torch.save(backbone, paths["vgg16"])
    torch.save(calibration, paths["lpips"])
    monkeypatch.setattr(weights, "_weight_paths", lambda run: paths)
    def forbid_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("Perceptual loaders must never download weights.")
    monkeypatch.setattr(weights, "_download", forbid_download)
    metric = weights.load_lpips(SimpleNamespace(), "cpu")
    assert not metric.training and all(not parameter.requires_grad for parameter in metric.parameters())
    torch.testing.assert_close(metric.net.slice1[0].weight, backbone["features.0.weight"])
    torch.testing.assert_close(metric.net.slice2[0].weight, backbone["features.4.weight"])
    for index, layer in enumerate(metric.lins):
        assert layer is getattr(metric, f"lin{index}")
        torch.testing.assert_close(layer.model[1].weight, calibration[f"lin{index}.model.1.weight"])


def test_perceptual_loader_requires_explicit_acquisition_receipt(tmp_path: Path) -> None:
    fake = SimpleNamespace(path=tmp_path / "run", root=tmp_path)
    with pytest.raises(FileNotFoundError, match="explicit acquire_weights"):
        weights._weight_paths(fake)
