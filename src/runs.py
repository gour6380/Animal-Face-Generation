"""Numbered comparison runs, immutable provenance, and explicit resume."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from .config import ExperimentConfig, config_from_dict
from .device import select_device
from .io import atomic_bytes, atomic_json, digest_object, read_json, sha256_file
from .locking import RunGate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_contents(root: Path) -> dict[str, bytes]:
    contents = {str(p.relative_to(root)): p.read_bytes() for p in sorted((root / "src").rglob("*.py"))}
    for name in ("requirements.in", "requirements.txt", ".python-version"):
        if (root / name).exists():
            contents[name] = (root / name).read_bytes()
    for notebook in sorted((root / "notebooks").glob("*.ipynb")):
        value = json.loads(notebook.read_text())
        value.get("metadata", {}).pop("widgets", None)
        for cell in value["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)
        contents[str(notebook.relative_to(root))] = json.dumps(value, sort_keys=True).encode()
    return contents


def source_identity(root: Path) -> str:
    import hashlib
    # The notebook is an execution interface: changing its explicit resume selector
    # or analysis controls must not invalidate training. Resolved training config
    # and all implementation source remain immutable; the original notebook is saved.
    return digest_object({name: hashlib.sha256(content).hexdigest() for name, content in _source_contents(root).items()
                          if not name.endswith(".ipynb")})


def environment_identity(device: str) -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "device": device, "dtype": "float32", "cpu_threads": torch.get_num_threads(),
        "packages": dict(sorted((d.metadata["Name"].lower(), d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))}


@dataclass
class Run:
    root: Path
    path: Path
    config: ExperimentConfig
    device: str
    hardware: dict[str, Any] | None = None
    _writer_lock: RunGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._writer_lock = RunGate(self.path / ".writer.lock")

    def lock(self) -> Any:
        return self._writer_lock.acquire()

    @contextmanager
    def training_lock(self, family: str) -> Iterator[None]:
        """Only distinct families may train concurrently on immutable inputs."""
        directory = self.model_dir(family)
        with self._writer_lock.acquire(shared=True), FileLock(directory / ".training.lock", timeout=0):
            yield

    def model_dir(self, family: str) -> Path:
        if family not in ("tokenizer", "diffusion", "autoregressive"):
            raise ValueError(f"Unknown model family: {family}")
        return self.path / "models" / family

    def verify(self) -> None:
        if torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != "cpu":
            raise RuntimeError("Restart the kernel with default float32 dtype and CPU allocation; model placement is explicit.")
        metadata = read_json(self.path / "run.json")
        if metadata["config_sha256"] != digest_object(self.config.to_dict()):
            raise ValueError("Run configuration changed; create a fresh run.")
        if metadata["config_sha256"] != digest_object(read_json(self.path / "config" / "experiment.json")):
            raise ValueError("Saved run configuration is corrupted.")
        if metadata["source_sha256"] != source_identity(self.root):
            raise ValueError("Source or requirements changed; create a fresh run.")
        for name, expected in metadata["snapshot_files"].items():
            if sha256_file(self.path / "source_snapshot" / name) != expected:
                raise ValueError(f"Saved source snapshot is corrupted: {name}")
        if read_json(self.path / "environment" / "environment.json") != metadata["environment"]:
            raise ValueError("Saved environment record is corrupted.")
        if metadata["environment"] != environment_identity(self.device):
            raise ValueError("Environment/backend changed; create a fresh run.")
        if str(select_device(self.config.runtime.device)) != self.device:
            raise ValueError("Device selection changed; create a fresh run.")
        if "hardware" in metadata:
            expected = metadata["hardware"]
            if digest_object(expected) != metadata["hardware_sha256"]:
                raise ValueError("Saved hardware plan is corrupted.")
            current = _hardware_plan(self.device, self.hardware)
            if current != expected:
                raise ValueError("Hardware/topology changed; create a fresh run.")
            self.hardware = current


def _hardware_plan(device: str, hardware: dict[str, Any] | None) -> dict[str, Any]:
    from .hardware import detect_hardware, validate_worker_binding

    inherited = os.environ.get("AFG_HARDWARE_JSON")
    proposed = hardware if hardware is not None else json.loads(inherited) if inherited else None
    if proposed is None:
        return detect_hardware(device)
    if proposed["device"] != device:
        raise ValueError("Hardware plan backend differs from run backend.")
    validate_worker_binding(proposed)
    return proposed


def create_run(project_root: str | Path, config: ExperimentConfig,
               hardware: dict[str, Any] | None = None) -> Run:
    root = Path(project_root).expanduser().resolve()
    config.validate()
    if not (root / "src" / "config.py").is_file() or not (root / "requirements.txt").is_file():
        raise ValueError("Select the animal-face-generation project root with its locked requirements.")
    torch.set_num_threads(config.runtime.cpu_threads)
    if torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != "cpu":
        raise RuntimeError("Restart the kernel with default float32 dtype and CPU allocation.")
    device = str(select_device(config.runtime.device))
    hardware = _hardware_plan(device, hardware)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    with FileLock(runs / ".allocation.lock", timeout=0):
        counter = runs / ".counter.json"
        numbers = [int(p.name[4:]) for p in runs.iterdir() if p.is_dir() and re.fullmatch(r"run_\d+", p.name)]
        number = max(numbers + [read_json(counter)["last"] if counter.exists() else 0]) + 1
        atomic_json(counter, {"last": number})
        path = runs / f"run_{number:03d}"
        path.mkdir()
    run = Run(root, path, config, device, hardware)
    for name in ("config", "environment", "source_snapshot", "data", "logs", "tensorboard", "evaluation", "figures", "tables", "report"):
        (path / name).mkdir()
    for family in ("tokenizer", "diffusion", "autoregressive"):
        run.model_dir(family).mkdir(parents=True)
    atomic_json(path / "config" / "experiment.json", config.to_dict())
    for name, content in _source_contents(root).items():
        atomic_bytes(path / "source_snapshot" / name, content)
    env = environment_identity(device)
    atomic_json(path / "environment" / "environment.json", env)
    atomic_json(path / "run.json", {"schema_version": 2, "created_at": utc_now(), "run_name": path.name,
        "config_sha256": digest_object(config.to_dict()), "source_sha256": source_identity(root), "environment": env,
        "hardware": hardware, "hardware_sha256": digest_object(hardware),
        "snapshot_files": {name: sha256_file(path / "source_snapshot" / name) for name in _source_contents(root)}})
    return run


def open_run(project_root: str | Path, run_name: str, config: ExperimentConfig | None = None,
             hardware: dict[str, Any] | None = None) -> Run:
    root = Path(project_root).expanduser().resolve()
    if not re.fullmatch(r"run_\d+", run_name):
        raise ValueError("Use an explicit run_XXX name.")
    path = root / "runs" / run_name
    saved_config = config_from_dict(read_json(path / "config" / "experiment.json"))
    if config is not None:
        proposed = {key: value for key, value in config.to_dict().items() if key != "evaluation"}
        original = {key: value for key, value in saved_config.to_dict().items() if key != "evaluation"}
        if digest_object(proposed) != digest_object(original):
            raise ValueError("Notebook model/data/runtime configuration differs from this run; create a fresh run.")
    # Preserve the original run configuration. A notebook's evaluation override is
    # passed separately to freeze_evaluation, which allocates an analysis revision.
    config = saved_config
    torch.set_num_threads(config.runtime.cpu_threads)
    device = read_json(path / "run.json")["environment"]["device"]
    hardware = _hardware_plan(device, hardware)
    run = Run(root, path, config, device, hardware)
    run.verify()
    return run


def model_checkpoint_identity(run: Run, family: str, kind: str = "final") -> str:
    return sha256_file(run.model_dir(family) / f"{kind}.pt")
