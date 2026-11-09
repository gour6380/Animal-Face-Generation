"""Synthetic tests for packed preprocessing, identity, and repeated preparation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src import data
from src.config import ExperimentConfig
from src.image_cache import open_snapshot, ordered_parallel_map
from src.io import atomic_json, digest_object


def make_run(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "project"
    path = root / "runs" / "run_001"
    (path / "data").mkdir(parents=True)
    config = ExperimentConfig()
    config = replace(config, data=replace(config.data, expected_total=42))
    run = SimpleNamespace(root=root, path=path, config=config,
                          lock=lambda: nullcontext(), verify=lambda: None)
    for partition, count in (("train", 10), ("val", 4)):
        for label, name in enumerate(config.data.classes):
            for index in range(count):
                destination = root / "data" / "raw" / "afhq_v2" / partition / name / f"{index:03d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Include high-frequency structure to make Lanczos equality meaningful.
                rows, columns = np.indices((512, 512))
                image = np.stack(((rows + index * 5) % 256, (columns + label * 17) % 256,
                                  (rows + columns + (100 if partition == "val" else 0)) % 256), axis=-1).astype(np.uint8)
                Image.fromarray(image).save(destination)
    atomic_json(path / "data" / "acquisition.json", {"dataset_root": "data/raw/afhq_v2"})
    return run


def test_packed_preprocessing_reads_raw_once_and_matches_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(tmp_path)
    original = Path.read_bytes
    reads = []

    def read(path: Path) -> bytes:
        if path.suffix == ".png":
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    inventory = data.validate_inventory(run)
    assert len(reads) == len(set(reads)) == 42
    row = inventory["records"][3]
    with Image.open(run.root / row["relative_path"]) as image:
        array = np.array(image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS), copy=True)
    expected = torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1)
    dataset = data.ImageDataset(run.root, [row])
    torch.testing.assert_close(dataset[0]["image"], expected, rtol=0, atol=0)
    assert len(reads) == 42
    assert len(dataset._snapshots) == 1
    dataset[0]
    assert len(dataset._snapshots) == 1


def test_packed_inputs_are_immutable_snapshot_of_verified_raw_bytes(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    inventory = data.validate_inventory(run)
    row = inventory["records"][0]
    dataset = data.ImageDataset(run.root, [row])
    expected = dataset[0]["image"]
    # The saved input is the immutable snapshot; unused raw changes cannot alter it.
    (run.root / row["relative_path"]).write_bytes(b"unused raw source changed")
    torch.testing.assert_close(dataset[0]["image"], expected, rtol=0, atol=0)
    assert "raw source hashes retained" in inventory["input_policy"]


def test_corrupt_packed_image_fails_at_consumption_and_inventory_reuse(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    inventory = data.validate_inventory(run)
    row = inventory["records"][0]
    array = np.load(run.root / inventory["snapshot"]["path"], mmap_mode="r+")
    array[0, 0, 0, 0] ^= 1
    array.flush()
    with pytest.raises(ValueError, match="Preprocessed image bytes changed"):
        data.ImageDataset(run.root, [row])[0]
    with pytest.raises(ValueError, match="snapshot checksum changed"):
        data.validate_inventory(run)


def test_snapshot_reused_across_runs_and_split_preparation_reads_no_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = make_run(tmp_path)
    inventory = data.validate_inventory(run)
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        if path.suffix == ".png":
            pytest.fail("Committed snapshots must not reread raw source files.")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = data.prepare_data(run)
    assert data.prepare_data(run) == manifest
    other_path = run.root / "runs" / "run_002"
    (other_path / "data").mkdir(parents=True)
    atomic_json(other_path / "data" / "acquisition.json", {"dataset_root": "data/raw/afhq_v2"})
    other = SimpleNamespace(**{**run.__dict__, "path": other_path})
    assert data.validate_inventory(other) == inventory
    assert data.prepare_data(other) == manifest


def test_snapshot_header_and_preprocessing_rejected_when_incompatible(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    inventory = data.validate_inventory(run)
    metadata = inventory["snapshot"]
    with pytest.raises(ValueError, match="shape, layout"):
        open_snapshot(run.root, {**metadata, "shape": [1, 128, 128, 3]})
    with pytest.raises(ValueError, match="image size"):
        data.ImageDataset(run.root, inventory["records"], image_size=64)[0]


def test_corrupt_original_image_prevents_snapshot_commit(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    path = run.root / "data/raw/afhq_v2/train/cat/000.png"
    path.write_bytes(b"invalid image")
    with pytest.raises(ValueError, match="Cannot validate image train/cat/000.png"):
        data.validate_inventory(run)
    assert not (run.path / "data/inventory.json").exists()
    assert not list((run.root / "data/preprocessed").glob("*/inventory.json"))


def test_shared_snapshot_identity_contains_preprocessing_version(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    inventory = data.validate_inventory(run)
    specification = inventory["snapshot_specification"]
    assert specification["version"] == 1
    assert specification["image_size"] == 128
    assert specification["resize"] == "PIL.LANCZOS"
    assert digest_object(specification) in inventory["snapshot"]["path"]


def test_ordered_parallel_map_preserves_order_and_propagates_failure() -> None:
    assert list(ordered_parallel_map(lambda value: value * 2, range(25), 4)) == list(range(0, 50, 2))

    def fail(value: int) -> int:
        if value == 5:
            raise ValueError("broken input")
        return value

    with pytest.raises(ValueError, match="broken input"):
        list(ordered_parallel_map(fail, range(25), 4))
