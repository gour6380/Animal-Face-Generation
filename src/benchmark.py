"""Synchronized, warmed end-to-end sampling benchmarks with explicit memory semantics."""
from __future__ import annotations

import platform
import threading
from time import perf_counter
from typing import Any, Self

import numpy as np
import torch
from tqdm.auto import tqdm

from .device import memory_snapshot, synchronize
from .evaluate import _analysis_path, load_freeze
from .io import atomic_json, digest_object, read_json
from .sample import load_generator, setting_spec


def benchmark_device_identity(run: Any) -> dict[str, Any]:
    """Record actual timing hardware and require the common first CUDA device."""
    from .hardware import cuda_device_uuid

    plan = getattr(run, "hardware", None) or {}
    identity = {"device": str(run.device), "hardware_plan_sha256": digest_object(plan)}
    if torch.device(run.device).type == "cuda":
        properties = torch.cuda.get_device_properties(torch.device(run.device))
        uuid = cuda_device_uuid(properties)
        selected = plan.get("selected_gpus", [])
        if selected and uuid != selected[0]["uuid"]:
            raise RuntimeError("All benchmarks must use the same first selected CUDA GPU.")
        identity["gpu"] = {"uuid": uuid, "name": properties.name,
                           "capability": [properties.major, properties.minor],
                           "total_memory": properties.total_memory}
    else:
        identity["host"] = plan.get("host", {"system": platform.system(), "machine": platform.machine()})
    return identity


class _MemorySampler:
    """Observed maxima, never advertised as exact MPS peaks or additive memory."""
    def __init__(self, device: str, interval: float = 0.05):
        self.device, self.interval = device, interval
        self.stop = threading.Event()
        self.maxima: dict[str, int | float] = {}
        self.minimum_available: int | None = None
        self.observations = 0
        self.error: str | None = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _record(self) -> None:
        row = memory_snapshot(self.device)
        self.observations += 1
        available = row["system_available_bytes"]
        self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "system_available_bytes":
                self.maxima[key] = max(self.maxima.get(key, value), value)

    def _sample(self) -> None:
        try:
            while not self.stop.is_set():
                self._record()
                self.stop.wait(self.interval)
        except (RuntimeError, OSError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> Self:
        self._record()
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join()
        self._record()

    def summary(self) -> dict[str, Any]:
        return {"observed_maxima": dict(sorted(self.maxima.items())),
                "minimum_system_available_bytes": self.minimum_available,
                "poll_interval_seconds": self.interval, "observations": self.observations,
                "polling_error": self.error,
                "meaning": "Sampled maxima, not exact peaks. Process RSS and MPS counters overlap in unified memory; do not add them."}


def benchmark_generation(run: Any, family: str, setting: str,
                         analysis: str = "analysis_001") -> dict[str, Any]:
    freeze = load_freeze(run, analysis)
    spec = setting_spec(family, setting)
    if freeze["settings"].get(setting) != spec:
        raise ValueError("Benchmark setting was not frozen.")
    cfg = freeze["evaluation_config"]
    identity = {"freeze_sha256": digest_object(freeze), "setting": spec, "device": str(run.device),
                "timing_hardware": benchmark_device_identity(run),
                "cpu_threads": torch.get_num_threads(), "platform": platform.platform(),
                "machine": platform.machine(), "torch_version": str(torch.__version__)}
    path = _analysis_path(run, analysis) / "benchmarks" / f"{setting}.json"
    with run.lock():
        if path.exists():
            existing = read_json(path)
            if existing["identity"] != identity:
                raise RuntimeError("Benchmark environment changed; use a new analysis revision.")
            return existing
        result: dict[str, Any] = {
            "identity": identity, "batches": {},
            "timing_includes": ["seeded CPU random sampling and device transfers", "prior sampling", "VQ lookup/snap",
                                "shared decoder", "uint8 conversion", "CPU output transfer", "device synchronization"],
            "timing_excludes": ["checkpoint/model loading", "warmups", "file IO", "feature extraction", "metrics", "plots"],
            "class_schedule": "cat/dog/wildlife in rotation; same class within each batch",
            "limitations": "Laptop power and thermal state are uncontrolled; memory polling adds small observer overhead.",
        }
        for batch in cfg["benchmark_batches"]:
            wrapper = load_generator(run, family)
            synchronize(run.device)
            cold_started = perf_counter()
            cold_images = wrapper.generate(run.config.data.classes[0], batch,
                                           cfg["sampling_seed"] + 700_000,
                                           steps=spec.get("steps", 50), temperature=spec.get("temperature", 1.0))
            synchronize(run.device)
            cold_seconds = perf_counter() - cold_started
            del cold_images
            for warmup in range(cfg["benchmark_warmups"]):
                wrapper.generate(run.config.data.classes[warmup % 3], batch,
                                 cfg["sampling_seed"] + 800_000 + warmup,
                                 steps=spec.get("steps", 50), temperature=spec.get("temperature", 1.0))
            synchronize(run.device)
            if torch.device(run.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(run.device))
            elapsed = []
            with _MemorySampler(str(run.device)) as observer:
                for repeat in tqdm(range(cfg["benchmark_repetitions"]), desc=f"Benchmark {setting}, batch {batch}", unit="trial"):
                    synchronize(run.device)
                    started = perf_counter()
                    images = wrapper.generate(run.config.data.classes[repeat % 3], batch,
                                              cfg["sampling_seed"] + 900_000 + repeat,
                                              steps=spec.get("steps", 50), temperature=spec.get("temperature", 1.0))
                    synchronize(run.device)
                    elapsed.append(perf_counter() - started)
                    del images
            metrics = {"batch_size": batch, "warmups": cfg["benchmark_warmups"],
                       "first_call_seconds": cold_seconds,
                       "first_call_meaning": "first generation call of a freshly constructed wrapper, not a fresh process or cold OS/device cache",
                       "repetitions": cfg["benchmark_repetitions"], "raw_seconds": elapsed,
                       "median_batch_seconds": float(np.median(elapsed)),
                       "p95_batch_seconds": float(np.percentile(elapsed, 95)),
                       "median_seconds_per_image": float(np.median(elapsed)) / batch,
                       "aggregate_images_per_second": batch * len(elapsed) / sum(elapsed),
                       "memory": observer.summary()}
            if torch.device(run.device).type == "cuda":
                metrics["memory"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(run.device))
            result["batches"][str(batch)] = metrics
            del wrapper
        atomic_json(path, result)
        return result
