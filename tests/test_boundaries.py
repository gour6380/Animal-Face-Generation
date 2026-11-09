"""Authored execution-boundary tests; intentionally not run at implementation handoff."""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src import benchmark, checkpoints, data, sample, tokenization
from src.config import EvaluationConfig, ExperimentConfig
from src.models import autoregressive, diffusion


def test_imports_and_architecture_inspection_do_not_download_train_or_create_runs(tmp_path):
    """Use a fresh interpreter so imports actually execute under the boundary guards."""
    project = Path(__file__).resolve().parents[1]
    script = r'''
import importlib
import os
from pathlib import Path
import socket
import sys
import urllib.request
import torch

root = Path(sys.argv[1]).resolve()

def blocked(*args, **kwargs):
    raise AssertionError("Import or inspection attempted network access or model execution")

socket.create_connection = blocked
socket.socket.connect = blocked
urllib.request.urlopen = blocked
torch.hub.load_state_dict_from_url = blocked
torch.nn.Module.__call__ = blocked
torch.optim.Adam.step = blocked
torch.optim.AdamW.step = blocked

# Even a broken import cannot create experiment or dataset artifacts during this test.
def audit(event, arguments):
    if event not in ("open", "os.mkdir", "os.remove", "os.rename"):
        return
    path = arguments[0]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(path)).resolve()
    if not any(path.is_relative_to(root / name) for name in ("data", "runs")):
        return
    if event == "open":
        flags = arguments[2] or 0
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            return
    raise AssertionError("Import or inspection attempted to mutate data/run artifacts")

sys.addaudithook(audit)
for module in ("config", "data", "device", "models.tokenizer", "models.diffusion",
               "models.autoregressive", "train", "tokenization", "sample", "evaluate",
               "benchmark", "checkpoints", "runs", "tracking", "reporting", "weights",
               "hardware", "launcher", "execution", "locking", "telemetry", "distributed"):
    importlib.import_module("src." + module)
from src.config import ExperimentConfig
from src.models.tokenizer import VQAutoencoder
config = ExperimentConfig()
config.validate()
model = VQAutoencoder(config.tokenizer)
assert "VQAutoencoder" in str(model)
assert sum(parameter.numel() for parameter in model.parameters()) > 0
'''
    environment = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1",
                   "MPLCONFIGDIR": str(tmp_path / "matplotlib"), "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-c", script, str(project)], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr


def test_benchmark_times_only_synchronized_generation_and_separates_cold_and_warm(tmp_path, monkeypatch):
    events: list[object] = []
    config = EvaluationConfig()
    freeze = {"settings": {"ddim_50": sample.setting_spec("diffusion", "ddim_50")},
              "evaluation_config": asdict(config)}
    run = SimpleNamespace(path=tmp_path, device="cpu", config=ExperimentConfig(), lock=nullcontext)
    monkeypatch.setattr(benchmark, "load_freeze", lambda run, analysis: freeze)
    monkeypatch.setattr(benchmark, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(benchmark, "synchronize", lambda device: events.append("sync"))

    clock = iter(index * 0.25 for index in range(1000))

    def read_clock():
        events.append("clock")
        return next(clock)

    class FakeGenerator:
        def generate(self, class_name, count, seed, **kwargs):
            events.append(("generate", count, seed, class_name))
            return object()

    def load_generator(run, family):
        events.append("load_model")
        return FakeGenerator()

    class FakeMemorySampler:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append("memory_start")
            return self

        def __exit__(self, *args):
            events.append("memory_stop")

        def summary(self):
            return {"observations": 1}

    monkeypatch.setattr(benchmark, "perf_counter", read_clock)
    monkeypatch.setattr(benchmark, "load_generator", load_generator)
    monkeypatch.setattr(benchmark, "_MemorySampler", FakeMemorySampler)
    monkeypatch.setattr(benchmark, "atomic_json", lambda path, value: events.append("write_results"))
    result = benchmark.benchmark_generation(run, "diffusion", "ddim_50")
    calls = [event for event in events if isinstance(event, tuple) and event[0] == "generate"]
    assert len(calls) == 2 * (1 + 3 + 30)
    assert events.count("load_model") == 2
    assert events[-1] == "write_results"
    for batch in (1, 8):
        metrics = result["batches"][str(batch)]
        assert metrics["warmups"] == 3 and metrics["repetitions"] == 30
        assert len(metrics["raw_seconds"]) == 30
        assert metrics["first_call_seconds"] == pytest.approx(0.25)
        assert metrics["median_batch_seconds"] == pytest.approx(0.25)
        assert metrics["aggregate_images_per_second"] == pytest.approx(batch / 0.25)
        batch_calls = [event for event in calls if event[1] == batch]
        assert [event[3] for event in batch_calls[4:7]] == ["cat", "dog", "wild"]
    # Every measured interval encloses generate and post-work synchronization.
    for index, event in enumerate(events):
        if isinstance(event, tuple) and event[0] == "generate":
            seed = event[2] - config.sampling_seed
            if seed == 700_000 or seed >= 900_000:
                assert events[index - 2:index] == ["sync", "clock"]
                assert events[index + 1:index + 3] == ["sync", "clock"]


def _fake_saved_generator(monkeypatch, tmp_path, family, *, randomized=False):
    """Mock only file/model boundaries; exercise the real Generator orchestration."""
    expected_ids = torch.arange(16 * 16).reshape(1, 16, 16) % 2
    codec_manifest = {"codec_sha256": "selected-codec", "mean": 2.0, "std": 4.0,
                      "token_cache_identity": "frozen-tokens"}
    observations = {"decode_calls": 0, "nearest_inputs": []}

    class FakeCodec(nn.Module):
        def lookup(self, ids):
            return (ids.float() * 2 - 1).unsqueeze(1)

        def nearest_codes(self, latent):
            observations["nearest_inputs"].append(latent.clone())
            return (latent[:, 0] > 0).long()

        def decode(self, latent):
            observations["decode_calls"] += 1
            return latent.expand(-1, 3, -1, -1)

    class FakePrior(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.register_buffer("anchor", torch.tensor(0.0))

        def generate(self, labels, generator, temperature, use_cache):
            if randomized:
                return torch.randint(2, (len(labels), 16, 16), generator=generator)
            return expected_ids.repeat(len(labels), 1, 1)

    class FakeSchedule(nn.Module):
        def __init__(self, *args):
            super().__init__()

        def ddim_sample(self, model, initial_noise, labels, steps):
            raw = (expected_ids.repeat(len(labels), 1, 1).float() * 2 - 1).unsqueeze(1)
            return (raw - codec_manifest["mean"]) / codec_manifest["std"]

    def dataset_access_forbidden(*args, **kwargs):
        raise AssertionError("Saved inference accessed dataset images or token caches")

    monkeypatch.setattr(data, "prepare_data", dataset_access_forbidden)
    monkeypatch.setattr(data, "ImageDataset", dataset_access_forbidden)
    monkeypatch.setattr(tokenization, "load_token_cache", dataset_access_forbidden)
    monkeypatch.setattr(tokenization, "load_frozen_tokenizer", lambda run, device: (FakeCodec(), codec_manifest))
    monkeypatch.setattr(checkpoints, "load_model_checkpoint", lambda run, family, kind: {
        "model": {"anchor": torch.tensor(0.0)}, "ema": {"anchor": torch.tensor(0.0)},
        "metadata": dict(codec_manifest),
    })
    monkeypatch.setattr(autoregressive, "ARTransformer", FakePrior)
    monkeypatch.setattr(diffusion, "DiffusionUNet", FakePrior)
    monkeypatch.setattr(diffusion, "DiffusionSchedule", FakeSchedule)
    run = SimpleNamespace(root=tmp_path, path=tmp_path / "run_001", device="cpu",
                          config=ExperimentConfig(), verify=lambda: None)
    return sample.load_generator(run, family), observations, expected_ids


@pytest.mark.parametrize("family", ["diffusion", "autoregressive"])
def test_saved_inference_uses_no_dataset_or_token_cache(tmp_path, monkeypatch, family):
    wrapper, observations, _ = _fake_saved_generator(monkeypatch, tmp_path, family)
    images = wrapper.generate("wildlife", count=2, seed=99)
    assert images.shape == (2, 3, 16, 16) and images.dtype == torch.uint8
    assert observations["decode_calls"] == 1
    assert not (tmp_path / "data").exists()


def test_sampling_rng_is_private_and_repeatable_after_global_rng_changes(tmp_path, monkeypatch):
    wrapper, _, _ = _fake_saved_generator(monkeypatch, tmp_path, "autoregressive", randomized=True)
    torch.manual_seed(65)
    before = torch.get_rng_state().clone()
    first = wrapper.generate("cat", count=2, seed=900)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(921)
    torch.randn(73)
    after = torch.get_rng_state().clone()
    second = wrapper.generate("cat", count=2, seed=900)
    different = wrapper.generate("cat", count=2, seed=901)
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_diffusion_inverts_normalization_once_then_uses_the_shared_decoder(tmp_path, monkeypatch):
    diffuse, observations, ids = _fake_saved_generator(monkeypatch, tmp_path, "diffusion")
    diffusion_images = diffuse.generate("dog", count=1, seed=71)
    expected_embedding = (ids.float() * 2 - 1).unsqueeze(1)
    assert len(observations["nearest_inputs"]) == 1
    torch.testing.assert_close(observations["nearest_inputs"][0], expected_embedding)
    autoreg, ar_observations, _ = _fake_saved_generator(monkeypatch, tmp_path, "autoregressive")
    ar_images = autoreg.generate("dog", count=1, seed=71)
    assert observations["decode_calls"] == ar_observations["decode_calls"] == 1
    assert torch.equal(diffusion_images, ar_images)
