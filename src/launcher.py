"""Standard-library notebook controller with isolated processes and serial queues."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WORKER_CODE = "from src.execution import worker_main; worker_main()"

def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Worker:
    """A durable job description with either a live child or a queued dependency."""

    label: str
    request_path: Path
    result_path: Path
    log_path: Path
    resource: tuple[str, ...]
    visible: str | None
    coordinator: bool = False
    predecessor: Worker | None = None
    process: subprocess.Popen | None = None
    cancelled: bool = False
    submitted: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    group_finalized: bool = False
    process_created: float | None = None

    def status(self) -> dict[str, Any]:
        code = self.process.poll() if self.process is not None else None
        if code is not None and self.finished is None:
            self.finished = time.monotonic()
        state = (
            "cancelled"
            if self.cancelled
            else "queued"
            if self.process is None
            else "running"
            if code is None
            else "complete"
            if code == 0
            else "failed"
        )
        return {
            "worker": self.label,
            "pid": self.process.pid if self.process else None,
            "status": state,
            "exit_code": code,
            "elapsed_seconds": round((self.finished or time.monotonic()) - self.started, 1) if self.started else 0,
            "queued_seconds": round((self.started or time.monotonic()) - self.submitted, 1),
            "resource": self.resource,
            "log": str(self.log_path),
            "result": str(self.result_path),
        }


class ExperimentLauncher:
    """Use separate CUDA devices concurrently; queue CPU/MPS/single-CUDA generators.

    Training models never enter the notebook process. OS leases survive notebook object
    resets while children run. A queued request starts only through this
    controller's wait(), and is never silently replayed after a kernel restart.
    """

    def __init__(self, project_root: str | Path, python_executable: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.python = Path(python_executable or sys.executable)
        if not self.python.is_file():
            raise FileNotFoundError(f"Python interpreter not found: {self.python}")
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.leases = self.root / ".device-locks"
        self.leases.mkdir(exist_ok=True)
        self.hardware: dict[str, Any] | None = None
        self._workers: list[Worker] = []
        atexit.register(self._shutdown)

    def _new_worker(
        self,
        request: dict,
        *,
        label: str,
        resource: tuple[str, ...],
        visible: str | None,
        coordinator: bool = False,
        predecessor: Worker | None = None,
    ) -> Worker:
        job_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{label}-{uuid.uuid4().hex[:8]}"
        directory = self.jobs / job_id
        directory.mkdir()
        worker = Worker(
            label,
            directory / "request.json",
            directory / "result.json",
            directory / "worker.log",
            resource,
            visible,
            coordinator,
            predecessor,
        )
        _atomic_json(worker.request_path, {**request, "project_root": str(self.root),
                                          "job_directory": str(directory)})
        worker.log_path.touch()
        self._workers.append(worker)
        if predecessor is None:
            self._spawn(worker)
        else:
            _atomic_json(directory / "process.json", worker.status())
            print(f"Queued {label} behind {predecessor.label}; device resources will not overlap.")
        return worker

    def _spawn(self, worker: Worker) -> None:
        leases = []
        try:
            gate = (self.leases / "execution.lock").open("a+")
            leases.append(gate)
            mode = fcntl.LOCK_EX if worker.coordinator else fcntl.LOCK_SH
            try:
                fcntl.flock(gate, mode | fcntl.LOCK_NB)
                for resource in worker.resource:
                    # Resource identifiers come from the validated hardware plan.
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", resource)
                    lease = (self.leases / f"{safe_name}.lock").open("a+")
                    leases.append(lease)
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Device resources are busy ({worker.resource or 'coordinator'}); wait or cancel the existing worker"
                ) from error
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env.update(
                PYTHONPATH=str(self.root),
                PYTHONUNBUFFERED="1",
                MPLBACKEND="Agg",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false",
                MPLCONFIGDIR=str(
                    Path(os.environ.get("TMPDIR", "/tmp")) / "animal-face-generation-matplotlib"
                ),
                AFG_REQUEST_PATH=str(worker.request_path),
                AFG_RESULT_PATH=str(worker.result_path),
                AFG_LEASE_FDS=",".join(str(lease.fileno()) for lease in leases),
            )
            request = json.loads(worker.request_path.read_text())
            if "hardware" in request:
                env["AFG_HARDWARE_JSON"] = json.dumps(request["hardware"])
                env["AFG_WORKER_ROLE"] = ("tokenizer" if request["action"] == "train_tokenizer"
                                          else request["action"].removeprefix("train_")
                                          if request["action"] in {"train_diffusion", "train_autoregressive"}
                                          else "coordinator")
            if worker.visible is not None:
                env["CUDA_VISIBLE_DEVICES"] = worker.visible
                env["AFG_WORKER_UUIDS"] = worker.visible
            else:
                env.pop("AFG_WORKER_UUIDS", None)
            with worker.log_path.open("ab", buffering=0) as log:
                worker.process = subprocess.Popen(
                    [str(self.python), "-u", "-c", _WORKER_CODE],
                    cwd=self.root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(lease.fileno() for lease in leases),
                )
            worker.started = time.monotonic()
            import psutil
            try:
                worker.process_created = psutil.Process(worker.process.pid).create_time()
            except psutil.NoSuchProcess:
                worker.process_created = None
            _atomic_json(worker.result_path.parent / "process.json", worker.status())
            print(f"Started {worker.label} (PID {worker.process.pid}). Log: {worker.log_path}")
        except BaseException:
            worker.cancelled = True
            if worker.process is not None and worker.process.poll() is None:
                try:
                    self.cancel(worker)
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"Worker cancellation bookkeeping failed: {error}", file=sys.stderr)
            else:
                _atomic_json(worker.result_path.parent / "process.json", worker.status())
            raise
        finally:
            for lease in leases:
                lease.close()  # The child keeps inherited leases until it exits.

    def inspect_hardware(self, requested: str = "auto") -> dict[str, Any]:
        """Discover devices in a fresh interpreter; no models or downloads."""
        return self.run_action("inspect_hardware", requested=requested)

    def _plan(self) -> dict[str, Any]:
        if self.hardware is None:
            raise RuntimeError("Run inspect_hardware() before creating or opening an experiment")
        return self.hardware

    def _assignment(self, family: str) -> tuple[tuple[str, ...], str]:
        plan = self._plan()
        if plan["device"] == "cuda":
            uuids = plan["assignments"][family]
            return tuple(f"cuda_{value}" for value in uuids), ",".join(uuids)
        return (plan["device"],), ""

    def run_action(self, action: str, run_name: str | None = None, **kwargs: Any) -> Any:
        """Execute exactly one explicit preparation, inspection or analysis operation."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", action):
            raise ValueError("Invalid action name")
        if any(worker.status()["status"] in {"running", "queued"} for worker in self._workers):
            raise RuntimeError("Wait for or cancel pending workers before coordinator actions")
        request = {"action": action, **kwargs}
        if run_name is not None:
            if not re.fullmatch(r"run_[0-9]{3,}", run_name):
                raise ValueError("Select an explicitly named run_XXX")
            request["run_name"] = run_name
        if action == "inspect_hardware":
            resource, visible = (), None
        else:
            request["hardware"] = self._plan()
            family = ("tokenizer" if action == "train_tokenizer" else "autoregressive"
                      if action == "train_autoregressive" else "diffusion")
            # All analysis is serial on GPU zero; timing never compares different GPUs.
            resource, visible = self._assignment(family)
            if action in {"create_run", "open_run"}:
                resource, visible = (), None
        worker = self._new_worker(request, label=action, resource=resource, visible=visible, coordinator=True)
        result = self.wait([worker])[action]
        if action == "inspect_hardware":
            from .hardware import validate_hardware_plan
            validate_hardware_plan(result)
            self.hardware = result
        return result

    def start_training(self, run_name: str, family: str) -> Worker:
        """Start one generator, or queue it behind the other on a shared device."""
        if family not in {"diffusion", "autoregressive"}:
            raise ValueError("family must be diffusion or autoregressive; tokenizer uses run_action")
        if not re.fullmatch(r"run_[0-9]{3,}", run_name):
            raise ValueError("Select an explicitly named run_XXX")
        plan = self._plan()
        pending = [worker for worker in self._workers if worker.status()["status"] in {"running", "queued"}]
        if any(worker.label == family for worker in pending):
            raise RuntimeError(f"The {family} model already has a pending worker")
        if any(worker.coordinator for worker in pending):
            raise RuntimeError("A coordinator action is already running")
        predecessor = pending[-1] if pending and not plan["parallel_generators"] else None
        resource, visible = self._assignment(family)
        return self._new_worker({"action": "train_" + family, "run_name": run_name,
                                 "family": family, "hardware": plan}, label=family,
                                resource=resource, visible=visible, predecessor=predecessor)

    @staticmethod
    def _tail(path: Path, size: int = 1500) -> str:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
        return "\n".join(lines[-4:])

    def _result(self, worker: Worker) -> Any:
        if worker.cancelled or worker.process is None or worker.process.poll() != 0:
            raise RuntimeError(
                f"{worker.label} failed or was cancelled. See {worker.log_path}\n{self._tail(worker.log_path)}"
            )
        if not worker.result_path.is_file():
            raise RuntimeError(
                f"{worker.label} exited without a committed result: {worker.log_path}"
            )
        value = json.loads(worker.result_path.read_text())
        if not value.get("ok"):
            raise RuntimeError(f"{worker.label}: {value}")
        if self._group_alive(worker):
            self.cancel(worker)
            raise RuntimeError(f"{worker.label} exited while descendant processes remained; they were stopped.")
        worker.group_finalized = True
        return value["result"]

    @staticmethod
    def _group_alive(worker: Worker) -> bool:
        """Include orphaned ranks after a coordinator dies; finalized jobs are inert."""
        if worker.group_finalized or worker.process is None:
            return False
        leader_exited = worker.process.poll() is not None
        if leader_exited and worker.process_created is not None:
            import psutil
            try:
                current_birth = psutil.Process(worker.process.pid).create_time()
            except psutil.NoSuchProcess:
                current_birth = None
            if current_birth is not None and current_birth != worker.process_created:
                # Compare creation times: a just-reaped leader may still briefly
                # appear in macOS process tables and must not be mistaken for reuse.
                worker.group_finalized = True
                return False
        try:
            os.killpg(worker.process.pid, 0)
        except ProcessLookupError:
            return False
        if leader_exited:
            # Some Pod PID-1 implementations leave orphaned zombies unreaped.
            # Zombies have released GPU/file descriptors and cannot be killed;
            # do not report failed cleanup solely because killpg(0) still sees them.
            import psutil
            observed_member = False
            try:
                for process in psutil.process_iter(["pid", "status"]):
                    try:
                        if os.getpgid(process.pid) == worker.process.pid:
                            observed_member = True
                            if process.info["status"] != psutil.STATUS_ZOMBIE:
                                return True
                    except ProcessLookupError:
                        continue
            except (PermissionError, psutil.AccessDenied):
                return True  # Restricted process listing must not prevent group signals.
            return not observed_member  # A racing/incomplete process snapshot cannot prove cleanup.
        return True

    def wait(self, workers: Sequence[Worker], poll_seconds: float = 1.0) -> dict[str, Any]:
        if not workers or not 0 < poll_seconds <= 30:
            raise ValueError("Supply workers and a polling interval in (0, 30] seconds")
        if len({worker.label for worker in workers}) != len(workers):
            raise ValueError("Worker labels must be distinct")
        if any(not any(worker is owned for owned in self._workers) for worker in workers):
            raise ValueError("Wait accepts only this controller's workers")
        watched = list(workers)
        for worker in list(watched):
            predecessor = worker.predecessor
            while predecessor is not None:
                if not any(predecessor is item for item in watched):
                    watched.append(predecessor)
                predecessor = predecessor.predecessor
        next_update = 0.0
        previous_tails: dict[Path, str] = {}
        try:
            while True:
                for worker in watched:
                    if worker.status()["status"] in {"failed", "cancelled"}:
                        self._result(worker)
                    if worker.process is None and not worker.cancelled:
                        predecessor = worker.predecessor
                        if predecessor is None or predecessor.status()["status"] == "complete":
                            if predecessor is not None:
                                self._result(predecessor)  # Require its committed success first.
                            self._spawn(worker)
                states = [worker.status() for worker in watched]
                if all(state["status"] == "complete" for state in states):
                    break
                if time.monotonic() >= next_update:
                    for worker, state in zip(watched, states, strict=True):
                        print(f"{worker.label}: {state['status']}, {state['elapsed_seconds']:.0f}s")
                        tail = self._tail(worker.log_path)
                        if tail and tail != previous_tails.get(worker.log_path):
                            print(tail)
                            previous_tails[worker.log_path] = tail
                    next_update = time.monotonic() + 30
                time.sleep(poll_seconds)
            result = {worker.label: self._result(worker) for worker in workers}
            for worker in watched:
                _atomic_json(worker.result_path.parent / "process.json", worker.status())
            return result
        except BaseException:
            self._cancel_many(watched, suppress_errors=True)
            raise

    def status(self) -> list[dict[str, Any]]:
        """Read state only; this never starts queued work."""
        return [worker.status() for worker in self._workers]

    def cancel(self, worker: Worker) -> None:
        if not any(worker is owned for owned in self._workers):
            raise ValueError("Cannot cancel another controller's worker")
        if worker.group_finalized:
            return
        if worker.process is None:
            worker.cancelled = True
        elif self._group_alive(worker):
            worker.cancelled = True
            # A dead coordinator can leave DDP ranks holding inherited GPU leases.
            # Signal the group even after Popen.poll() has reaped its leader.
            for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 4)):
                try:
                    os.killpg(worker.process.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + timeout
                while self._group_alive(worker) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not self._group_alive(worker):
                    break
            if self._group_alive(worker):
                raise RuntimeError(f"Worker process group did not exit: {worker.process.pid}")
        worker.group_finalized = True
        _atomic_json(worker.result_path.parent / "process.json", worker.status())

    def _cancel_many(self, workers: Sequence[Worker], *, suppress_errors: bool) -> None:
        first_error = None
        for worker in workers:
            try:
                self.cancel(worker)
            except BaseException as error:  # noqa: BLE001 — try every child before propagating failure.
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

    def _shutdown(self) -> None:
        self._cancel_many(self._workers, suppress_errors=True)

    def cancel_all(self) -> None:
        """Attempt every cancellation, then report any bookkeeping or process error."""
        self._cancel_many(self._workers, suppress_errors=False)
