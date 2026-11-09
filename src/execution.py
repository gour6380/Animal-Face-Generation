"""Internal isolated-worker dispatch. Notebook calls are the public interface."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_ACTIONS = {
    "download_dataset": "data", "validate_inventory": "data", "prepare_data": "data",
    "acquire_weights": "weights", "describe_model": "inspection", "train_tokenizer": "train",
    "train_diffusion": "train", "train_autoregressive": "train",
    "evaluate_reconstructions": "tokenization", "freeze_tokenizer": "tokenization", "cache_tokens": "tokenization",
    "freeze_evaluation": "evaluate", "prepare_reference": "evaluate", "evaluate_samples": "evaluate",
    "generate_samples": "sample", "generate_ddpm_examples": "sample", "generate_diffusion_process": "sample",
    "generate_autoregressive_process": "sample", "benchmark_generation": "benchmark",
    "write_report": "reporting", "show_training_samples": "reporting", "plot_training_history": "reporting",
    "show_reconstructions": "reporting", "show_generated_grid": "reporting", "plot_comparison": "reporting",
    "plot_generation_process": "reporting", "show_nearest_training": "reporting", "show_ddpm_examples": "reporting",
}


def _stop_children(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def _distributed_tokenizer(request: dict[str, Any]) -> Any:
    """Two separately masked ranks; one failed rank cancels its peers promptly."""
    from .io import atomic_json
    directory = Path(request["job_directory"])
    uuids = request["hardware"]["assignments"]["tokenizer"]
    children: list[subprocess.Popen] = []
    result_paths = []
    rendezvous = (directory / "distributed-rendezvous").resolve().as_uri()
    try:
        for rank, uuid in enumerate(uuids):
            rank_directory = directory / f"rank_{rank}"
            rank_directory.mkdir()
            request_path, result_path = rank_directory / "request.json", rank_directory / "result.json"
            atomic_json(request_path, {**request, "action": "tokenizer_rank", "job_directory": str(rank_directory),
                                      "rank": rank, "world_size": len(uuids), "init_method": rendezvous})
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=uuid, AFG_WORKER_UUIDS=uuid, AFG_WORKER_ROLE="tokenizer",
                               AFG_REQUEST_PATH=str(request_path), AFG_RESULT_PATH=str(result_path))
            inherited = tuple(int(value) for value in environment.get("AFG_LEASE_FDS", "").split(",") if value)
            children.append(subprocess.Popen([sys.executable, "-u", "-c",
                                               "from src.execution import worker_main; worker_main()"],
                                              cwd=request["project_root"], env=environment, pass_fds=inherited))
            result_paths.append(result_path)
        while True:
            codes = [child.poll() for child in children]
            failures = [index for index, code in enumerate(codes) if code is not None and code != 0]
            if failures:
                raise RuntimeError(f"Tokenizer rank(s) {failures} failed; stopping all ranks. See worker log and rank result files.")
            if all(code == 0 for code in codes):
                break
            time.sleep(0.2)
        results = [json.loads(path.read_text()) for path in result_paths]
        if not all(result.get("ok") for result in results):
            raise RuntimeError("Tokenizer rank exited without a successful committed result.")
        return results[0]["result"]
    finally:
        _stop_children(children)


def execute(request: dict[str, Any]) -> Any:
    """Dispatch a closed set of explicitly requested notebook operations."""
    action = request["action"]
    if action == "inspect_hardware":
        from .hardware import detect_hardware
        return detect_hardware(request.get("requested", "auto"))
    if action not in _ACTIONS and action not in {"create_run", "open_run", "tokenizer_rank", "inference"}:
        raise ValueError(f"Unknown notebook operation: {action}")
    from .config import EvaluationConfig, config_from_dict
    from .runs import create_run, open_run
    root = request["project_root"]
    hardware = request["hardware"]
    if action in {"create_run", "open_run"}:
        config = config_from_dict(request["config"]) if request.get("config") is not None else None
        if action == "create_run":
            if config is None:
                raise ValueError("create_run requires the explicit notebook configuration")
            run = create_run(root, config, hardware=hardware)
        else:
            run = open_run(root, request["run_name"], config=config, hardware=hardware)
        return {"run_name": run.path.name, "path": str(run.path), "device": run.device, "hardware": hardware}
    run = open_run(root, request["run_name"], hardware=hardware)
    if action == "tokenizer_rank":
        from .train import train_tokenizer_distributed
        return train_tokenizer_distributed(run, request["rank"], request["world_size"],
                                           request["init_method"], backend="nccl")
    if action == "train_tokenizer" and hardware["tokenizer_world_size"] > 1:
        return _distributed_tokenizer(request)
    reserved = {"action", "project_root", "job_directory", "run_name", "hardware"}
    kwargs = {key: value for key, value in request.items() if key not in reserved}
    if action.startswith("train_"):
        kwargs.pop("family", None)
    if action == "freeze_evaluation" and kwargs.get("evaluation_config") is not None:
        raw = dict(kwargs["evaluation_config"])
        for key in ("ddim_steps", "temperatures", "benchmark_batches"):
            if key in raw:
                raw[key] = tuple(raw[key])
        kwargs["evaluation_config"] = EvaluationConfig(**raw)
    if action == "inference":
        from PIL import Image

        from .sample import load_generator
        family = kwargs.pop("family")
        model = load_generator(run, family)
        images = model.generate(**kwargs)
        paths = []
        for index, tensor in enumerate(images):
            path = Path(request["job_directory"]) / f"inference_{index:03d}.png"
            Image.fromarray(tensor.permute(1, 2, 0).numpy()).save(path)
            paths.append(str(path))
        return {"family": family, "images": paths}
    function = getattr(importlib.import_module(f"src.{_ACTIONS[action]}"), action)
    result = function(run, **kwargs)
    if hasattr(result, "savefig"):
        path = Path(request["job_directory"]) / "figure.png"
        result.savefig(path, dpi=140, bbox_inches="tight")
        return {"figure": str(path)}
    return str(result) if isinstance(result, Path) else result


def worker_main() -> None:
    """Environment-only worker entrypoint; deliberately no experiment CLI arguments."""
    from .io import atomic_json
    from .telemetry import StageMonitor
    request_path, result_path = Path(os.environ["AFG_REQUEST_PATH"]), Path(os.environ["AFG_RESULT_PATH"])
    request = json.loads(request_path.read_text())
    directory = Path(request["job_directory"])
    uuids = [value for value in os.environ.get("AFG_WORKER_UUIDS", "").split(",") if value]
    monitor = StageMonitor(directory, request["action"], uuids)
    try:
        with monitor:
            value = execute(request)
        result = {"ok": True, "result": value}
    except BaseException as error:  # noqa: BLE001 — persist failure details even for interruption.
        traceback.print_exc()
        result = {"ok": False, "error": type(error).__name__, "message": str(error)}
    result["observations"] = monitor.summary
    atomic_json(result_path, result)
    if request.get("run_name") and request["action"] != "tokenizer_rank":
        record = Path(request["project_root"]) / "runs" / request["run_name"] / "logs" / "workers" / f"{directory.name}.json"
        atomic_json(record, {"action": request["action"], "result_file": str(result_path),
                             "telemetry_file": str(directory / "telemetry.jsonl"), **monitor.summary})
    if not result["ok"]:
        raise SystemExit(1)
