"""Best-effort stage observations; never alter execution to fit a resource target."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Self


def gpu_observations(uuids: list[str]) -> dict[str, Any]:
    """NVIDIA counters are per-device, including other users of that device."""
    if not uuids:
        return {}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=3, check=True).stdout
        values = {}
        for line in output.splitlines():
            uuid, utilization, used, total = (value.strip() for value in line.split(","))
            if uuid in uuids:
                values[uuid] = {"utilization_percent": float(utilization), "used_memory_mib": float(used),
                                "total_memory_mib": float(total), "scope": "whole GPU, not this process only"}
        return values
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"unavailable": type(error).__name__}


class StageMonitor:
    """Observe this process and descendants without timeout warnings or interruption."""
    def __init__(self, directory: Path, label: str, uuids: list[str], interval: float = 5.0):
        self.directory, self.label, self.uuids, self.interval = directory, label, uuids, interval
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.summary: dict[str, Any] = {}
        self.samples: list[dict[str, Any]] = []
        self._processes: dict[int, Any] = {}
        self.thread = threading.Thread(target=self._loop, name="stage-observations", daemon=True)

    def _snapshot(self) -> dict[str, Any]:
        import psutil
        current = psutil.Process(os.getpid())
        processes = [current] + current.children(recursive=True)
        rss, cpu = 0, 0.0
        for process in processes:
            try:
                known = self._processes.setdefault(process.pid, process)
                rss += known.memory_info().rss
                cpu += known.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {"elapsed_seconds": time.monotonic() - self.started, "process_tree_rss_bytes": rss,
                "process_tree_cpu_percent": cpu, "cpu_percent_meaning": "100 percent equals one logical CPU",
                "system_available_bytes": psutil.virtual_memory().available,
                "system_swap_bytes": psutil.swap_memory().used, "cuda": gpu_observations(self.uuids)}

    def _loop(self) -> None:
        with (self.directory / "telemetry.jsonl").open("a") as stream:
            while True:
                try:
                    value = self._snapshot()
                except Exception as error:  # noqa: BLE001 — observations must never stop a stage.
                    value = {"elapsed_seconds": time.monotonic() - self.started, "observation_error": repr(error)}
                self.samples.append(value)
                stream.write(json.dumps(value) + "\n")
                stream.flush()
                if self.stop.wait(self.interval):
                    break

    def __enter__(self) -> Self:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *exception: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        rows = [row for row in self.samples if "process_tree_rss_bytes" in row]
        self.summary = {"stage": self.label, "duration_seconds": time.monotonic() - self.started,
                        "samples": len(rows), "observed_peak_tree_rss_bytes": max(
                            (row["process_tree_rss_bytes"] for row in rows), default=None),
                        "mean_tree_cpu_percent": sum(row["process_tree_cpu_percent"] for row in rows) / len(rows) if rows else None,
                        "status": "failed" if exception and exception[0] else "complete",
                        "meaning": "Sampled observations, not exact peaks; child RSS can include shared pages."}
        (self.directory / "telemetry_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
