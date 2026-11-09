"""Explicit AFHQ-v2 acquisition, split provenance, and deterministic images."""
from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from PIL import __version__ as pillow_version
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .image_cache import (
    SNAPSHOT_VERSION,
    open_snapshot,
    ordered_parallel_map,
    verified_snapshot_row,
)
from .io import atomic_json, digest_object, immutable_json, read_json, sha256_file
from .runs import Run

LOGGER = logging.getLogger(__name__)


def _download(url: str, destination: Path) -> None:
    """Streaming explicit download. No automatic invocation from imports."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AnimalFaceGeneration/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0")) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    progress.update(len(block))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def _ensure_archive(run: Run, archive: Path) -> dict[str, Any]:
    """Keep a completed download across extraction failures, including old retries."""
    identity_path = archive.with_suffix(".download.json")
    previous = read_json(identity_path) if identity_path.exists() else None
    if previous is not None and previous["url"] != run.config.data.url:
        raise ValueError("Cached AFHQ archive belongs to a different download URL.")
    cached = archive.exists()
    if not cached:
        _download(run.config.data.url, archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("Cached AFHQ file is not a valid ZIP archive; inspect or remove it before retrying.")
    checksum = sha256_file(archive)
    if previous is not None:
        if previous["archive_sha256"] != checksum:
            raise ValueError("Cached AFHQ archive changed since its completed download.")
        return previous
    identity = {"url": run.config.data.url, "archive_sha256": checksum,
                "archive_bytes": archive.stat().st_size,
                "observed_identity_basis": "existing completed ZIP" if cached else "explicit download",
                "checksum_basis": "Locally observed SHA-256; not an independently supplied publisher checksum."}
    atomic_json(identity_path, identity)
    return identity


def _dataset_layout(archive: Path) -> tuple[PurePosixPath, str]:
    """Recognize original train/test (v2) or train/val layouts without renaming."""
    directories: set[tuple[str, ...]] = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = PurePosixPath(member.filename).parts
            if any(part == "__MACOSX" or part.startswith(".") for part in parts):
                continue
            end = len(parts) + 1 if member.is_dir() else len(parts)
            directories.update(parts[:length] for length in range(1, end))
    candidates = [(parts[:-1], held_out)
                  for parts in directories if parts[-1] == "train"
                  for held_out in ("test", "val") if parts[:-1] + (held_out,) in directories]
    if len(candidates) != 1:
        roots = sorted("/".join(parts) or "." for parts, _ in candidates)
        top_levels = sorted(parts[0] for parts in directories if len(parts) == 1)
        raise ValueError(f"Expected one AFHQ root with train/ and test/ or val/; "
                         f"found {len(candidates)} layouts {roots}, top-level folders {top_levels}.")
    parts, held_out = candidates[0]
    return PurePosixPath(*parts), held_out


def download_dataset(run: Run) -> dict[str, Any]:
    with run.lock(), FileLock(run.root / "data" / ".acquisition.lock", timeout=0):
        run.verify()
        root = run.root
        receipt_path = root / "data" / "acquisition.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["url"] != run.config.data.url or sha256_file(root / receipt["archive"]) != receipt["archive_sha256"]:
                raise ValueError("Cached AFHQ download identity changed.")
            if not (root / receipt["dataset_root"]).is_dir():
                raise FileNotFoundError("Cached extraction is missing; repair the data cache explicitly.")
        else:
            archive = root / "data" / "raw" / "afhq_v2.zip"
            download = _ensure_archive(run, archive)
            archive_root, held_out = _dataset_layout(archive)
            parent = archive.parent
            target = parent / "afhq_v2"
            if target.exists():
                raise FileExistsError("Unregistered extraction already exists; inspect it before retrying.")
            temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=parent))
            try:
                safe_extract(archive, temporary)
                shutil.move(str(temporary / archive_root), str(target))
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            receipt = {"dataset": "AFHQ v2", "url": run.config.data.url,
                "source": "https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq",
                "license": "CC BY-NC 4.0", "archive": str(archive.relative_to(root)),
                "archive_sha256": download["archive_sha256"], "archive_bytes": download["archive_bytes"],
                "archive_root": str(archive_root), "partitions": {"train": "train", "held_out": held_out},
                "dataset_root": str(target.relative_to(root)),
                "checksum_basis": "Locally observed SHA-256 on first acquisition; not an independently supplied publisher checksum."}
            atomic_json(receipt_path, receipt)
        immutable_json(run.path / "data" / "acquisition.json", receipt)
        return receipt


def _verify_inventory(value: dict[str, Any], receipt: dict[str, Any]) -> None:
    if value.get("digest") != digest_object({key: row for key, row in value.items() if key != "digest"}):
        raise ValueError("Inventory was modified.")
    if value["acquisition_sha256"] != digest_object(receipt):
        raise ValueError("Inventory acquisition identity changed.")


def _inventory_rows(run: Run, dataset_root: Path, partitions: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for partition in (partitions["train"], partitions["held_out"]):
        actual_classes = sorted(p.name for p in (dataset_root / partition).iterdir() if p.is_dir())
        if actual_classes != sorted(run.config.data.classes):
            raise ValueError(f"Unexpected {partition} class directories: {actual_classes}")
        for label, name in enumerate(run.config.data.classes):
            paths = sorted(p for p in (dataset_root / partition / name).rglob("*")
                           if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if not paths:
                raise ValueError(f"Empty dataset partition: {partition}/{name}")
            records.extend({"id": str(path.relative_to(dataset_root)), "relative_path": str(path.relative_to(run.root)),
                            "label": label, "class_name": name, "source_split": partition} for path in paths)
    if len(records) != run.config.data.expected_total:
        raise ValueError(f"Expected AFHQ v2 total {run.config.data.expected_total}, found {len(records)}. "
                         "Inspect provenance before changing config.")
    return records


def validate_inventory(run: Run) -> dict[str, Any]:
    """Validate raw inputs once and commit their immutable RGB128 snapshots.

    The packed snapshot is the input actually consumed by later stages. Its
    per-image hashes are checked on every read, independent of unused raw files.
    """
    with run.lock():
        run.verify()
        receipt = read_json(run.path / "data" / "acquisition.json")
        partitions = receipt.get("partitions", {"train": "train", "held_out": "val"})
        specification = {"version": SNAPSHOT_VERSION, "acquisition_sha256": digest_object(receipt),
                         "classes": list(run.config.data.classes), "expected_total": run.config.data.expected_total,
                         "image_size": run.config.data.image_size, "resize": "PIL.LANCZOS", "mode": "RGB",
                         "pillow_version": pillow_version}
        cache_parent = run.root / "data" / "preprocessed"
        cache_parent.mkdir(parents=True, exist_ok=True)
        cache_root = cache_parent / digest_object(specification)
        with FileLock(cache_parent / ".inventory.lock", timeout=0):
            shared_inventory = cache_root / "inventory.json"
            if shared_inventory.exists():
                result = read_json(shared_inventory)
                _verify_inventory(result, receipt)
                if result["snapshot_specification"] != specification:
                    raise ValueError("Preprocessed inventory specification changed.")
                snapshot = result["snapshot"]
                open_snapshot(run.root, snapshot)
                LOGGER.info("Verifying cached packed image snapshot (%s bytes)", snapshot["bytes"])
                if sha256_file(run.root / snapshot["path"]) != snapshot["file_sha256"]:
                    raise ValueError("Preprocessed snapshot checksum changed; repair the cache explicitly.")
                immutable_json(run.path / "data" / "inventory.json", result)
                return result
            if cache_root.exists():
                raise ValueError("Incomplete preprocessed cache exists; inspect it before retrying.")
            records = _inventory_rows(run, run.root / receipt["dataset_root"], partitions)
            size = run.config.data.image_size
            shape = (len(records), size, size, 3)
            temporary = Path(tempfile.mkdtemp(prefix=".preprocess-", dir=cache_parent))
            snapshot_path = str((cache_root / "images.npy").relative_to(run.root))

            def inspect(row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
                path = run.root / row["relative_path"]
                contents = path.read_bytes()
                try:
                    with Image.open(BytesIO(contents)) as source:
                        source.load()
                        if source.size != (512, 512):
                            raise ValueError(f"Unexpected AFHQv2 dimensions: {path}: {source.size}")
                        rgb = source.convert("RGB")
                        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                        resized = rgb.resize((size, size), Image.Resampling.LANCZOS)
                        array = np.array(resized, dtype=np.uint8, copy=True)
                except (OSError, ValueError) as error:
                    raise ValueError(f"Cannot validate image {row['id']}: {error}") from error
                return {**row, "file_sha256": hashlib.sha256(contents).hexdigest(), "pixel_sha256": pixel_hash}, array

            try:
                packed = np.lib.format.open_memmap(temporary / "images.npy", mode="w+", dtype=np.uint8, shape=shape)
                seen: dict[str, tuple[str, int]] = {}
                completed = []
                workers = run.config.runtime.io_workers
                inspected = ordered_parallel_map(inspect, records, workers)
                for index, (row, array) in enumerate(tqdm(inspected, total=len(records), desc="Validate + pack RGB images")):
                    identity = (row["source_split"], row["label"])
                    if row["pixel_sha256"] in seen and seen[row["pixel_sha256"]] != identity:
                        raise ValueError("Exact duplicate pixels span upstream partitions or labels; inspect dataset integrity.")
                    seen[row["pixel_sha256"]] = identity
                    packed[index] = array
                    row["snapshot"] = {"path": snapshot_path, "index": index,
                                       "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                                       "shape": list(shape), "dtype": "uint8", "version": SNAPSHOT_VERSION}
                    completed.append(row)
                packed.flush()
                del packed
                with (temporary / "images.npy").open("rb") as stream:
                    os.fsync(stream.fileno())
                LOGGER.info("Hashing packed image snapshot")
                snapshot = {"path": snapshot_path, "shape": list(shape), "dtype": "uint8", "version": SNAPSHOT_VERSION,
                            "bytes": (temporary / "images.npy").stat().st_size,
                            "file_sha256": sha256_file(temporary / "images.npy")}
                result = {"acquisition_sha256": digest_object(receipt), "partitions": partitions,
                          "records": completed, "count": len(completed), "snapshot": snapshot,
                          "snapshot_specification": specification,
                          "input_policy": "Immutable RGB snapshot; raw source hashes retained; snapshot row hashes checked at consumption."}
                result["digest"] = digest_object(result)
                atomic_json(temporary / "inventory.json", result)
                os.replace(temporary, cache_root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            immutable_json(run.path / "data" / "inventory.json", result)
            return result


def prepare_data(run: Run) -> dict[str, Any]:
    """Create immutable development splits after explicit inventory validation."""
    with run.lock():
        run.verify()
        if (run.path / "data" / "manifest.json").exists():
            return load_prepared_data(run)
        inventory = read_json(run.path / "data" / "inventory.json")
        original_digest = inventory["digest"]
        if digest_object({k: v for k, v in inventory.items() if k != "digest"}) != original_digest:
            raise ValueError("Inventory was modified.")
        records = inventory["records"]
        receipt = read_json(run.path / "data" / "acquisition.json")
        _verify_inventory(inventory, receipt)
        partitions = receipt.get("partitions", {"train": "train", "held_out": "val"})
        if "snapshot" in inventory:
            open_snapshot(run.root, inventory["snapshot"])
        else:
            # Older inventories have no immutable pixel snapshot.
            for row in tqdm(records, desc="Verify split source files"):
                if sha256_file(run.root / row["relative_path"]) != row["file_sha256"]:
                    raise ValueError(f"Dataset file changed: {row['id']}")
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        rng = random.Random(run.config.data.split_seed)
        for label in range(len(run.config.data.classes)):
            rows = sorted((r for r in records if r["source_split"] == partitions["train"] and r["label"] == label), key=lambda r: r["id"])
            rng.shuffle(rows)
            count = max(1, int(len(rows) * run.config.data.validation_fraction))
            splits["validation"].extend(rows[:count])
            splits["train"].extend(rows[count:])
            splits["test"].extend(r for r in records if r["source_split"] == partitions["held_out"] and r["label"] == label)
        seen: dict[str, str] = {}
        for name, rows in splits.items():
            rows.sort(key=lambda r: r["id"])
            for row in rows:
                previous = seen.get(row["pixel_sha256"])
                if previous is not None and previous != name:
                    raise ValueError("Exact duplicated pixels cross derived splits. Review duplicates; do not train.")
                seen[row["pixel_sha256"]] = name
        n = min(run.config.evaluation.max_per_class,
            min(sum(r["label"] == label for r in splits["test"]) for label in range(3)))
        if n < 4:
            raise ValueError("Insufficient held-out examples for the registered metrics.")
        result = {"schema_version": 1, "dataset_root": receipt["dataset_root"], "upstream_partitions": partitions,
            "inventory_sha256": original_digest, "splits": splits, "n_per_class": n,
            "preprocessing": {"image_size": 128, "rgb": True, "resize": "PIL.LANCZOS", "range": [-1, 1], "augmentation": None},
            "split_seed": run.config.data.split_seed,
            "test_role": f"upstream {partitions['held_out']} is held-out final test"}
        result["digest"] = digest_object(result)
        immutable_json(run.path / "data" / "manifest.json", result)
        return result


def load_prepared_data(run: Run) -> dict[str, Any]:
    """Read verified shared inputs without taking a coordinator lock or writing.

    Preparation belongs to an explicit notebook stage. Training workers cannot
    create or reconcile shared files while another family is training.
    """
    manifest = read_json(run.path / "data" / "manifest.json")
    inventory = read_json(run.path / "data" / "inventory.json")
    acquisition = read_json(run.path / "data" / "acquisition.json")
    for label, value in (("split manifest", manifest), ("inventory", inventory)):
        if value.get("digest") != digest_object({k: v for k, v in value.items() if k != "digest"}):
            raise ValueError(f"Prepared {label} was modified; rerun explicit preparation in a fresh run.")
    if manifest["inventory_sha256"] != inventory["digest"]:
        raise ValueError("Prepared inventory and split manifest differ.")
    if inventory["acquisition_sha256"] != digest_object(acquisition):
        raise ValueError("Prepared acquisition identity differs from the inventory.")
    if manifest["split_seed"] != run.config.data.split_seed:
        raise ValueError("Prepared split seed differs from the configuration.")
    if "snapshot" in inventory:
        open_snapshot(run.root, inventory["snapshot"])
    return manifest


class ImageDataset(Dataset):
    def __init__(self, root: str | Path, records: list[dict[str, Any]], image_size: int = 128):
        self.root = Path(root)
        self.records = records
        self.image_size = image_size
        self._snapshots: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        if "snapshot" in row:
            metadata = row["snapshot"]
            if metadata["shape"][1:] != [self.image_size, self.image_size, 3]:
                raise ValueError("Preprocessed image size differs from the requested preprocessing.")
            if metadata["path"] not in self._snapshots:
                self._snapshots[metadata["path"]] = open_snapshot(self.root, metadata)
            array = verified_snapshot_row(self._snapshots[metadata["path"]], metadata["index"],
                                          metadata["sha256"], row["id"])
        else:
            contents = (self.root / row["relative_path"]).read_bytes()
            if hashlib.sha256(contents).hexdigest() != row["file_sha256"]:
                raise ValueError(f"Image bytes changed since inventory: {row['id']}")
            with Image.open(BytesIO(contents)) as source:
                image = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1)
        return {"image": tensor, "label": torch.tensor(row["label"], dtype=torch.long), "id": row["id"]}
