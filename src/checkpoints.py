"""State-dict checkpoints with atomic completed-epoch commits and recovery."""
from __future__ import annotations

import csv
import io
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import atomic_bytes, atomic_json, digest_object, read_json, sha256_file
from .runs import Run


def capture_rng(device: str | torch.device | None = None) -> dict[str, Any]:
    """Capture CPU streams plus the selected accelerator (all when unspecified)."""
    state = np.random.get_state()
    backend = torch.device(device).type if device is not None else None
    result: dict[str, Any] = {"python": random.getstate(), "numpy": {"name": state[0],
        "keys": torch.tensor(state[1].astype(np.int64)), "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch": torch.get_rng_state()}
    if backend in (None, "mps") and torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    if backend in (None, "cuda") and torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], value["keys"].numpy().astype(np.uint32), value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return value


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_cpu_tree(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _identity(run: Run, family: str) -> dict[str, str]:
    return {"family": family, "run_name": run.path.name, "config_sha256": digest_object(run.config.to_dict()),
        "source_sha256": read_json(run.path / "run.json")["source_sha256"],
        "data_manifest_sha256": sha256_file(run.path / "data" / "manifest.json")}


def save_architecture(run: Run, family: str, model: torch.nn.Module) -> dict[str, Any]:
    value = {"family": family, "implementation_version": "1.0", "model_config": run.config.to_dict()[family],
        "image_size": run.config.data.image_size, "latent_size": run.config.latent_size,
        "codebook_size": run.config.tokenizer.codebook_size, "embedding_dim": run.config.tokenizer.embedding_dim,
        "classes": run.config.data.display_classes, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": read_json(run.path / "run.json")["source_sha256"]}
    directory = run.model_dir(family)
    path = directory / "architecture.json"
    if path.exists() and digest_object(read_json(path)) != digest_object(value):
        raise ValueError("Architecture changed within an existing run.")
    atomic_json(path, value)
    atomic_bytes(directory / "architecture.txt", (str(model) + f"\n\nParameters: {value['total_parameters']}\n").encode())
    return value


def _write_metrics(directory: Path, records: list[dict[str, Any]]) -> None:
    import json
    atomic_bytes(directory / "metrics.jsonl", "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records).encode())
    if records:
        out = io.StringIO()
        fields = ["epoch"] + sorted(set().union(*(r.keys() for r in records)) - {"epoch"})
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        atomic_bytes(directory / "metrics.csv", out.getvalue().encode())
    else:
        atomic_bytes(directory / "metrics.csv", b"")


def _records(directory: Path, epoch: int) -> list[dict[str, Any]]:
    import json
    path = directory / "metrics.jsonl"
    if not path.exists():
        return []
    return [value for line in path.read_text().splitlines() if line and (value := json.loads(line))["epoch"] <= epoch]


def _copy_atomic(source: Path, target: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=".export-", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, name)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def _materialize(directory: Path, pointer: dict[str, Any],
                 kinds: tuple[str, ...] = ("last", "best_validation", "final")) -> None:
    # A single epoch can back all three names; verify its bytes only once here.
    verified: set[tuple[str, str]] = set()
    for kind in kinds:
        reference = pointer.get(kind)
        if reference is None:
            continue
        source = directory / reference["path"]
        identity = (reference["path"], reference["sha256"])
        if identity not in verified:
            if sha256_file(source) != reference["sha256"]:
                raise ValueError(f"Committed checkpoint corrupted: {source}")
            verified.add(identity)
        target = directory / f"{kind}.pt"
        if not target.exists() or sha256_file(target) != reference["sha256"]:
            _copy_atomic(source, target)


def save_epoch(run: Run, family: str, payload: dict[str, Any], metrics: dict[str, Any], is_best: bool, final: bool) -> Path:
    directory = run.model_dir(family)
    epoch = int(payload["epoch"])
    if epoch < 1:
        raise ValueError("Epoch records use completed one-based epoch numbers.")
    pointer_path = directory / "commit.json"
    previous = read_json(pointer_path) if pointer_path.exists() else {"epoch": 0}
    if previous.get("final"):
        raise ValueError("This training stage is complete; create a fresh run to extend training.")
    if epoch != int(previous["epoch"]) + 1:
        raise ValueError("Commit epochs consecutively after restoring the last completed epoch.")
    payload = dict(payload)
    payload["metadata"] = {**payload.get("metadata", {}), **_identity(run, family)}
    checkpoint = directory / "checkpoints" / f"epoch_{epoch:06d}.pt"
    atomic_torch_save(checkpoint, payload)
    reference = {"path": str(checkpoint.relative_to(directory)), "sha256": sha256_file(checkpoint)}
    records = _records(directory, int(previous["epoch"]))
    records.append({**metrics, "epoch": epoch})
    _write_metrics(directory, records)
    pointer = {"epoch": epoch, "last": reference,
        "best_validation": reference if is_best else previous.get("best_validation"),
        "final": reference if final else previous.get("final")}
    atomic_json(pointer_path, pointer)
    _materialize(directory, pointer)
    atomic_json(directory / "status.json", {"status": "complete" if final else "training", "epoch": epoch,
        "global_step": payload["global_step"], "last_sha256": reference["sha256"]})
    retained = {v["path"] for k, v in pointer.items() if k != "epoch" and v}
    if previous.get("last"):
        retained.add(previous["last"]["path"])
    for old in (directory / "checkpoints").glob("epoch_*.pt"):
        if str(old.relative_to(directory)) not in retained:
            old.unlink()
    return directory / "last.pt"


def load_training_state(run: Run, family: str) -> dict[str, Any] | None:
    directory = run.model_dir(family)
    pointer_path = directory / "commit.json"
    if not pointer_path.exists():
        _write_metrics(directory, [])
        return None
    pointer = read_json(pointer_path)
    _materialize(directory, pointer)
    records = _records(directory, int(pointer["epoch"]))
    if [r["epoch"] for r in records] != list(range(1, int(pointer["epoch"]) + 1)):
        raise ValueError("Committed metrics are incomplete or duplicated.")
    _write_metrics(directory, records)
    return load_model_checkpoint(run, family, "last")


def read_training_state(run: Run, family: str) -> dict[str, Any] | None:
    """Read a reconciled epoch without writing; used by nonzero DDP ranks.

    Rank zero must finish ``load_training_state`` before peers enter here.
    The authoritative committed file is checked instead of repairing exports.
    """
    directory = run.model_dir(family)
    pointer_path = directory / "commit.json"
    if not pointer_path.exists():
        return None
    pointer = read_json(pointer_path)
    reference = pointer["last"]
    source = directory / reference["path"]
    if sha256_file(source) != reference["sha256"]:
        raise ValueError(f"Committed checkpoint corrupted: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    expected = _identity(run, family)
    if any(value.get("metadata", {}).get(key) != v for key, v in expected.items()):
        raise ValueError("Checkpoint run/source/config/data identity mismatch.")
    if value["epoch"] != pointer["epoch"]:
        raise ValueError("Checkpoint epoch disagrees with its commit record.")
    return value


def load_model_checkpoint(run: Run, family: str, kind: str = "final") -> dict[str, Any]:
    if kind not in ("last", "best_validation", "final"):
        raise ValueError("Unknown checkpoint selection.")
    directory = run.model_dir(family)
    pointer = read_json(directory / "commit.json")
    if not pointer.get(kind):
        raise ValueError(f"No committed {kind} checkpoint for {family}.")
    # Inference needs one committed checkpoint, not every training/recovery view.
    _materialize(directory, pointer, (kind,))
    value = torch.load(directory / f"{kind}.pt", map_location="cpu", weights_only=True)
    expected = _identity(run, family)
    if any(value.get("metadata", {}).get(key) != v for key, v in expected.items()):
        raise ValueError("Checkpoint run/source/config/data identity mismatch.")
    return value
