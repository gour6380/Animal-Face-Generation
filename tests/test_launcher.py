"""Offline worker lifecycle tests: no GPUs, datasets, model weights or training."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from src.hardware import build_hardware_plan
from src.launcher import ExperimentLauncher


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    package = tmp_path / "src"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "execution.py").write_text('''
import json, os, time
from pathlib import Path

def worker_main():
    request = json.loads(Path(os.environ["AFG_REQUEST_PATH"]).read_text())
    started = time.time()
    time.sleep(request.get("duration", 0.3))
    failure = request.get("run_name") == "run_999"
    result = {"ok": not failure, "result": {"started": started, "ended": time.time(),
              "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "role": os.environ.get("AFG_WORKER_ROLE"), "action": request["action"],
              "device_order": os.environ.get("CUDA_DEVICE_ORDER")}}
    Path(os.environ["AFG_RESULT_PATH"]).write_text(json.dumps(result))
    if failure:
        raise SystemExit(1)
''')
    return tmp_path


def plan(device: str = "cpu", gpu_count: int = 0) -> dict:
    gpus = [{"index": index, "uuid": f"GPU-fake-{index}", "name": "Synthetic CUDA",
             "capability": [12, 0], "total_memory": 24 * 1024**3} for index in range(gpu_count)]
    return build_hardware_plan(device, gpus)


def launcher(root: Path, hardware: dict | None = None) -> ExperimentLauncher:
    value = ExperimentLauncher(root, sys.executable)
    value.hardware = hardware or plan()
    return value


def test_single_device_queues_and_does_not_start_from_status(fake_project: Path) -> None:
    controller = launcher(fake_project)
    first = controller.start_training("run_001", "diffusion")
    second = controller.start_training("run_001", "autoregressive")
    assert first.status()["status"] == "running"
    assert second.status()["status"] == "queued"
    controller.status()
    assert second.process is None
    output = controller.wait([first, second], poll_seconds=0.02)
    assert output["autoregressive"]["started"] >= output["diffusion"]["ended"]
    assert all(item["status"] == "complete" for item in controller.status())
    frozen = first.status()["elapsed_seconds"]
    time.sleep(0.03)
    assert first.status()["elapsed_seconds"] == frozen


def test_two_cuda_workers_are_concurrent_and_uuid_isolated(fake_project: Path) -> None:
    controller = launcher(fake_project, plan("cuda", 3))
    first = controller.start_training("run_001", "diffusion")
    second = controller.start_training("run_001", "autoregressive")
    assert first.process is not None and second.process is not None
    output = controller.wait([first, second], poll_seconds=0.02)
    left, right = output["diffusion"], output["autoregressive"]
    assert left["visible"] == "GPU-fake-0"
    assert right["visible"] == "GPU-fake-1"
    assert max(left["started"], right["started"]) < min(left["ended"], right["ended"])


def test_tokenizer_coordinator_leases_both_selected_gpus(fake_project: Path) -> None:
    controller = launcher(fake_project, plan("cuda", 3))
    output = controller.run_action("train_tokenizer", "run_001")
    assert output["visible"] == "GPU-fake-0,GPU-fake-1"
    assert output["role"] == "tokenizer"
    assert controller._workers[-1].resource == ("cuda_GPU-fake-0", "cuda_GPU-fake-1")


def test_duplicate_family_and_coordinator_overlap_are_rejected(fake_project: Path) -> None:
    controller = launcher(fake_project)
    worker = controller.start_training("run_001", "diffusion")
    with pytest.raises(RuntimeError, match="pending worker"):
        controller.start_training("run_001", "diffusion")
    with pytest.raises(RuntimeError, match="Wait for or cancel"):
        controller.run_action("prepare_data", "run_001")
    controller.wait([worker], poll_seconds=0.02)


def test_second_controller_cannot_reuse_live_device_and_leases_release(fake_project: Path) -> None:
    first = launcher(fake_project)
    second = launcher(fake_project)
    worker = first.start_training("run_001", "diffusion")
    with pytest.raises(RuntimeError, match="Device resources are busy"):
        second.start_training("run_001", "diffusion")
    first.cancel(worker)
    replacement = second.start_training("run_001", "diffusion")
    second.wait([replacement], poll_seconds=0.02)


def test_failed_predecessor_cancels_queue_and_leases_release(fake_project: Path) -> None:
    controller = launcher(fake_project)
    first = controller.start_training("run_999", "diffusion")
    second = controller.start_training("run_001", "autoregressive")
    with pytest.raises(RuntimeError, match="failed or was cancelled"):
        controller.wait([first, second], poll_seconds=0.02)
    assert second.status()["status"] == "cancelled"
    assert second.process is None
    replacement = controller.start_training("run_001", "diffusion")
    controller.wait([replacement], poll_seconds=0.02)


def test_wait_can_include_only_queued_worker_and_resolves_predecessor(fake_project: Path) -> None:
    controller = launcher(fake_project)
    controller.start_training("run_001", "diffusion")
    second = controller.start_training("run_001", "autoregressive")
    result = controller.wait([second], poll_seconds=0.02)
    assert result["autoregressive"]["action"] == "train_autoregressive"


def test_long_stage_has_no_timeout_warning_or_cancellation(fake_project: Path, capsys: pytest.CaptureFixture) -> None:
    controller = launcher(fake_project)
    worker = controller.start_training("run_001", "diffusion")
    worker.started -= 601
    output = controller.wait([worker], poll_seconds=0.02)
    assert output["diffusion"]["action"] == "train_diffusion"
    assert not worker.cancelled
    assert "ALERT:" not in capsys.readouterr().out
    status = json.loads((worker.result_path.parent / "process.json").read_text())
    assert status["status"] == "complete"


def test_distributed_coordinator_terminates_failed_rank_peers(fake_project: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real process cleanup with a fake rank backend, without Torch/DDP."""
    from src.execution import _distributed_tokenizer
    (fake_project / "src" / "execution.py").write_text('''
import json, os, time
from pathlib import Path

def worker_main():
    request = json.loads(Path(os.environ["AFG_REQUEST_PATH"]).read_text())
    Path(request["job_directory"], "pid").write_text(str(os.getpid()))
    if request["rank"] == 0:
        time.sleep(0.2)
        raise SystemExit(1)
    time.sleep(10)
    Path(os.environ["AFG_RESULT_PATH"]).write_text(json.dumps({"ok": True, "result": {}}))
''')
    directory = fake_project / "jobs" / "test_ddp_failure"
    directory.mkdir(parents=True)
    monkeypatch.delenv("AFG_LEASE_FDS", raising=False)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="failed; stopping all ranks"):
        _distributed_tokenizer({"project_root": str(fake_project), "job_directory": str(directory),
                                "hardware": plan("cuda", 2), "run_name": "run_001"})
    assert time.monotonic() - started < 5
    assert not (directory / "rank_1" / "result.json").exists()
    import psutil
    for pid_file in directory.glob("rank_*/pid"):
        assert not psutil.pid_exists(int(pid_file.read_text()))


def test_analysis_settings_always_use_first_gpu(fake_project: Path) -> None:
    controller = launcher(fake_project, plan("cuda", 2))
    value = controller.run_action("benchmark_generation", "run_001", family="autoregressive")
    assert value["visible"] == "GPU-fake-0"
    assert value["role"] == "coordinator"


def test_dead_coordinator_orphans_are_cancelled_and_leases_released(fake_project: Path) -> None:
    import os
    import signal

    import psutil
    (fake_project / "src" / "execution.py").write_text('''
import json, os, subprocess, sys, time
from pathlib import Path

def worker_main():
    request = json.loads(Path(os.environ["AFG_REQUEST_PATH"]).read_text())
    if request["run_name"] == "run_666":
        inherited = tuple(int(value) for value in os.environ["AFG_LEASE_FDS"].split(","))
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=inherited)
        Path(request["job_directory"], "child.pid").write_text(str(child.pid))
        time.sleep(30)
    else:
        Path(os.environ["AFG_RESULT_PATH"]).write_text(json.dumps({"ok": True, "result": {"complete": True}}))
''')
    controller = launcher(fake_project)
    worker = controller.start_training("run_666", "diffusion")
    child_path = worker.request_path.parent / "child.pid"
    deadline = time.monotonic() + 5
    while not child_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_path.exists()
    child_pid = int(child_path.read_text())
    os.kill(worker.process.pid, signal.SIGKILL)
    worker.process.wait(timeout=5)
    with pytest.raises(RuntimeError, match="failed or was cancelled"):
        controller.wait([worker], poll_seconds=0.02)
    deadline = time.monotonic() + 2
    while True:
        try:
            child_status = psutil.Process(child_pid).status()
        except psutil.NoSuchProcess:
            child_status = None
        if child_status in (None, psutil.STATUS_ZOMBIE) or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    assert child_status in (None, psutil.STATUS_ZOMBIE)
    assert worker.group_finalized
    # Repeating cancellation of historical jobs cannot signal a recycled PGID.
    controller.cancel(worker)
    replacement = controller.start_training("run_001", "diffusion")
    assert controller.wait([replacement], poll_seconds=0.02)["diffusion"]["complete"]


@pytest.mark.parametrize("order", [None, "FASTEST_FIRST", "PCI_BUS_ID"])
def test_worker_preserves_parent_cuda_order(fake_project: Path, monkeypatch: pytest.MonkeyPatch,
                                          order: str | None) -> None:
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    controller = launcher(fake_project, plan("cuda", 2))
    value = controller.run_action("describe_model", "run_001", family="diffusion")
    assert value["device_order"] == order


@pytest.mark.parametrize(("status", "live"), [("zombie", False), ("running", True)])
def test_orphan_zombie_group_is_not_a_live_resource_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         status: str, live: bool) -> None:
    import os
    from types import SimpleNamespace

    import psutil

    from src.launcher import Worker

    def missing_leader(pid: int, signal: int) -> None:
        raise ProcessLookupError

    worker = Worker("synthetic", tmp_path / "request.json", tmp_path / "result.json",
                    tmp_path / "worker.log", ("cpu",), "")
    worker.process = SimpleNamespace(pid=12345, poll=lambda: -9)
    monkeypatch.setattr(os, "kill", missing_leader)
    monkeypatch.setattr(os, "killpg", lambda pid, signal: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: 12345)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [SimpleNamespace(pid=54321, info={"status": status})])
    assert ExperimentLauncher._group_alive(worker) is live


def test_empty_process_snapshot_cannot_prove_orphan_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from types import SimpleNamespace

    import psutil

    from src.launcher import Worker

    def missing_leader(pid: int, signal: int) -> None:
        raise ProcessLookupError

    worker = Worker("synthetic", tmp_path / "request.json", tmp_path / "result.json",
                    tmp_path / "worker.log", ("cpu",), "")
    worker.process = SimpleNamespace(pid=12345, poll=lambda: -9)
    monkeypatch.setattr(os, "kill", missing_leader)
    monkeypatch.setattr(os, "killpg", lambda pid, signal: None)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [])
    assert ExperimentLauncher._group_alive(worker)


def test_restricted_process_listing_still_allows_group_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from types import SimpleNamespace

    import psutil

    from src.launcher import Worker

    def restricted_listing(attrs: list) -> None:
        raise PermissionError("Process enumeration denied")

    worker = Worker("synthetic", tmp_path / "request.json", tmp_path / "result.json",
                    tmp_path / "worker.log", ("cpu",), "")
    worker.process = SimpleNamespace(pid=12345, poll=lambda: -9)
    monkeypatch.setattr(os, "killpg", lambda pid, signal: None)
    monkeypatch.setattr(psutil, "process_iter", restricted_listing)
    assert ExperimentLauncher._group_alive(worker)
