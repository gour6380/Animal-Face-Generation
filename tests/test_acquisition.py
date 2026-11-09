"""AFHQ release-layout and interrupted-acquisition regressions; authored, not run."""
from __future__ import annotations

import zipfile
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from PIL import Image

from src import data
from src.config import ExperimentConfig
from src.io import atomic_json, digest_object, read_json, sha256_file


def _run(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "project"
    path = root / "runs" / "run_001"
    (root / "data" / "raw").mkdir(parents=True)
    (path / "data").mkdir(parents=True)
    return SimpleNamespace(root=root, path=path, config=ExperimentConfig(),
                           lock=lambda: nullcontext(), verify=lambda: None)


def _write_archive(path: Path, *, prefix: str = "", holdout: str = "test") -> None:
    """No explicit ZIP directory members: directory discovery must use ancestry."""
    with zipfile.ZipFile(path, "w") as archive:
        for partition in ("train", holdout):
            for animal in ("cat", "dog", "wild"):
                name = f"{prefix}{partition}/{animal}/00000.png"
                archive.writestr(name, name.encode())


@pytest.mark.parametrize("prefix", ["", "afhq_v2/", "release/afhq_v2/"])
@pytest.mark.parametrize("holdout", ["test", "val"])
def test_dataset_layout_preserves_actual_upstream_holdout(
    tmp_path: Path, prefix: str, holdout: str,
) -> None:
    archive = tmp_path / "afhq_v2.zip"
    _write_archive(archive, prefix=prefix, holdout=holdout)
    assert data._dataset_layout(archive) == (PurePosixPath(prefix or "."), holdout)


def test_dataset_layout_ignores_macos_metadata_tree(tmp_path: Path) -> None:
    archive = tmp_path / "afhq_v2.zip"
    _write_archive(archive)
    with zipfile.ZipFile(archive, "a") as source:
        source.writestr("__MACOSX/afhq/train/cat/._00000.png", b"metadata")
        source.writestr("__MACOSX/afhq/val/cat/._00000.png", b"metadata")
    assert data._dataset_layout(archive) == (PurePosixPath("."), "test")


@pytest.mark.parametrize("extra_prefix", ["", "second/"])
def test_dataset_layout_rejects_ambiguous_holdout_or_dataset_root(
    tmp_path: Path, extra_prefix: str,
) -> None:
    archive = tmp_path / "afhq_v2.zip"
    _write_archive(archive)
    with zipfile.ZipFile(archive, "a") as source:
        source.writestr(f"{extra_prefix}val/cat/example.png", b"second holdout")
        if extra_prefix:
            source.writestr(f"{extra_prefix}train/cat/example.png", b"second train")
    with pytest.raises(ValueError):
        data._dataset_layout(archive)


def test_dataset_layout_rejects_missing_heldout_partition(tmp_path: Path) -> None:
    archive = tmp_path / "afhq_v2.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("train/cat/example.png", b"training only")
    with pytest.raises(ValueError):
        data._dataset_layout(archive)


@pytest.mark.parametrize("holdout", ["test", "val"])
def test_acquisition_reuses_existing_archive_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, holdout: str,
) -> None:
    run = _run(tmp_path)
    archive = run.root / "data" / "raw" / "afhq_v2.zip"
    _write_archive(archive, prefix="afhq_v2/", holdout=holdout)

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("A completed archive must not be downloaded again.")

    monkeypatch.setattr(data, "_download", unexpected_download)
    first = data.download_dataset(run)
    second = data.download_dataset(run)
    assert first == second
    assert first["partitions"] == {"train": "train", "held_out": holdout}
    assert first["archive_sha256"] == sha256_file(archive)
    assert (run.root / first["dataset_root"] / holdout / "cat" / "00000.png").is_file()
    assert read_json(run.path / "data" / "acquisition.json") == first
    assert archive.with_suffix(".download.json").is_file()


def test_extraction_failure_retains_download_identity_and_reuses_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path)
    archive = run.root / "data" / "raw" / "afhq_v2.zip"
    _write_archive(archive)
    original_hash = sha256_file(archive)
    original_extract = data.safe_extract

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("Extraction recovery must reuse the completed ZIP.")

    def interrupted_extract(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic interrupted extraction")

    monkeypatch.setattr(data, "_download", unexpected_download)
    monkeypatch.setattr(data, "safe_extract", interrupted_extract)
    with pytest.raises(OSError, match="synthetic interrupted"):
        data.download_dataset(run)
    assert archive.with_suffix(".download.json").is_file()
    assert not (run.root / "data" / "acquisition.json").exists()
    assert sha256_file(archive) == original_hash
    monkeypatch.setattr(data, "safe_extract", original_extract)
    receipt = data.download_dataset(run)
    assert receipt["archive_sha256"] == original_hash
    assert receipt["partitions"]["held_out"] == "test"


def test_inventory_reads_original_test_directory_and_records_its_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path)
    run.config = replace(run.config, data=replace(run.config.data, expected_total=6))
    archive = run.root / "data" / "raw" / "afhq_v2.zip"
    with zipfile.ZipFile(archive, "w") as source:
        for partition, green in (("train", 50), ("test", 150)):
            for label, animal in enumerate(run.config.data.classes):
                buffer = BytesIO()
                Image.new("RGB", (512, 512), (label * 60, green, 10)).save(buffer, format="PNG")
                source.writestr(f"{partition}/{animal}/00000.png", buffer.getvalue())

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("Inventory validation must use the completed local archive.")

    monkeypatch.setattr(data, "_download", unexpected_download)
    receipt = data.download_dataset(run)
    inventory = data.validate_inventory(run)
    assert inventory["count"] == 6
    assert inventory["partitions"] == {"train": "train", "held_out": "test"}
    assert inventory["acquisition_sha256"] == digest_object(receipt)
    assert sum(row["source_split"] == "test" for row in inventory["records"]) == 3
    assert all(row["relative_path"].startswith("data/raw/afhq_v2/") for row in inventory["records"])


def test_invalid_cached_archive_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path)
    archive = run.root / "data" / "raw" / "afhq_v2.zip"
    original_bytes = b"incomplete or invalid archive"
    archive.write_bytes(original_bytes)

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("An invalid completed cache file requires explicit repair.")

    monkeypatch.setattr(data, "_download", unexpected_download)
    with pytest.raises(ValueError):
        data.download_dataset(run)
    assert archive.read_bytes() == original_bytes


@pytest.mark.parametrize("identity_change", ["url", "archive"])
def test_completed_download_identity_cannot_change_unnoticed(
    tmp_path: Path, identity_change: str,
) -> None:
    run = _run(tmp_path)
    archive = run.root / "data" / "raw" / "afhq_v2.zip"
    _write_archive(archive)
    data._ensure_archive(run, archive)
    if identity_change == "url":
        run.config = replace(run.config, data=replace(run.config.data, url="https://example.invalid/other.zip"))
    else:
        with zipfile.ZipFile(archive, "a") as source:
            source.writestr("unexpected.txt", b"archive was changed")
    with pytest.raises(ValueError):
        data._ensure_archive(run, archive)


@pytest.mark.parametrize("holdout", ["test", "val"])
def test_actual_upstream_holdout_is_excluded_from_development_splits(
    tmp_path: Path, holdout: str,
) -> None:
    run = _run(tmp_path)
    records = []
    for label, animal in enumerate(run.config.data.classes):
        for partition, count in (("train", 10), (holdout, 4)):
            for index in range(count):
                identity = f"{partition}/{animal}/{index:03d}.png"
                relative = f"data/raw/afhq_v2/{identity}"
                path = run.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(identity.encode())
                checksum = sha256_file(path)
                records.append({"id": identity, "relative_path": relative,
                    "label": label, "class_name": animal, "source_split": partition,
                    "file_sha256": checksum, "pixel_sha256": checksum})
    acquisition = {"dataset_root": "data/raw/afhq_v2",
                   "partitions": {"train": "train", "held_out": holdout}}
    inventory = {"acquisition_sha256": digest_object(acquisition),
                 "records": records, "count": len(records)}
    inventory["digest"] = digest_object(inventory)
    atomic_json(run.path / "data" / "acquisition.json", acquisition)
    atomic_json(run.path / "data" / "inventory.json", inventory)
    manifest = data.prepare_data(run)
    assert manifest == data.prepare_data(run)
    assert manifest["n_per_class"] == 4
    assert len(manifest["splits"]["test"]) == 12
    assert all(row["source_split"] == holdout for row in manifest["splits"]["test"])
    assert all(row["source_split"] == "train"
               for split in ("train", "validation") for row in manifest["splits"][split])
    identities = {split: {row["id"] for row in rows} for split, rows in manifest["splits"].items()}
    assert identities["train"].isdisjoint(identities["validation"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["validation"].isdisjoint(identities["test"])
