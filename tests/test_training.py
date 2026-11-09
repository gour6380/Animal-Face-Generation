"""Authored correctness tests. They are intentionally unexecuted at handoff."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, TrainingConfig
from src.io import atomic_json, sha256_file
from src.tokenization import latent_moments, load_token_cache
from src.train import (
    _assert_mutable,
    _loader,
    _loss_tokenizer,
    _noise,
    _run_epoch,
    lr_multiplier,
    microbatches,
    update_ema,
)


def test_microbatch_preserves_short_remainder():
    batch = {"label": torch.arange(7), "x": torch.arange(7), "id": [str(i) for i in range(7)]}
    portions = list(microbatches(batch, 3))
    assert [len(item["label"]) for item in portions] == [3, 3, 1]
    assert torch.equal(torch.cat([item["x"] for item in portions]), batch["x"])


def test_accumulation_weights_actual_samples_including_partial_effective_batch():
    # Ten samples => one effective batch of seven followed by a three-sample batch.
    data = [{"x": torch.tensor([float(i)]), "target": torch.tensor([float(i % 3)]),
             "label": torch.tensor(0)} for i in range(10)]
    model = nn.Linear(1, 1, bias=False)
    reference = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.2)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    config = replace(TrainingConfig(), effective_batch=7, microbatch=3, gradient_clip=1_000_000)

    def loss_function(batch):
        loss = (model(batch["x"]) - batch["target"]).square().mean()
        return loss, {"loss": loss.detach().item()}

    loader = DataLoader(data, batch_size=7)
    _, steps = _run_epoch(model, loader, config, loss_function, optimizer, None, 0, "test")
    for batch in DataLoader(data, batch_size=7):
        reference_optimizer.zero_grad()
        (reference(batch["x"]) - batch["target"]).square().mean().backward()
        reference_optimizer.step()
    assert steps == 2
    torch.testing.assert_close(model.weight, reference.weight)


def test_data_order_does_not_depend_on_global_model_rng():
    data = [{"label": i} for i in range(29)]
    config = replace(TrainingConfig(), effective_batch=8)
    first = torch.cat([batch["label"] for batch in _loader(data, config, 3, True)])
    torch.manual_seed(9128)
    torch.randn(400)
    second = torch.cat([batch["label"] for batch in _loader(data, config, 3, True)])
    next_epoch = torch.cat([batch["label"] for batch in _loader(data, config, 4, True)])
    assert torch.equal(first, second)
    assert not torch.equal(first, next_epoch)


def test_private_noise_does_not_advance_dropout_stream():
    torch.manual_seed(25)
    before = torch.get_rng_state().clone()
    noise1 = _noise(torch.Size([2, 3, 4]), torch.Generator().manual_seed(41), "cpu")
    noise2 = _noise(torch.Size([2, 3, 4]), torch.Generator().manual_seed(41), "cpu")
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(noise1, noise2)


def test_lr_has_five_warmup_epochs_and_exact_final_floor():
    config = replace(TrainingConfig(), epochs=200, warmup_epochs=5)
    assert lr_multiplier(0, config) == 0.2
    assert lr_multiplier(4, config) == 1.0
    assert lr_multiplier(5, config) == 1.0
    assert lr_multiplier(config.epochs - 1, config) == pytest.approx(0.1)
    assert lr_multiplier(80, replace(config, warmup_epochs=0)) == 1.0


def test_ema_updates_float_and_copies_integer_buffers():
    model = nn.BatchNorm1d(2)
    ema = {name: value.clone() for name, value in model.state_dict().items()}
    model.weight.data.fill_(3)
    model.num_batches_tracked.fill_(17)
    update_ema(ema, model, 0.5)
    torch.testing.assert_close(ema["weight"], torch.full((2,), 2.0))
    assert ema["num_batches_tracked"].item() == 17
    # CPU checkpoint roundtrip preserves every EMA key without replacing raw weights.
    restored = {name: value.clone() for name, value in ema.items()}
    model.load_state_dict(restored)
    assert model.weight[0].item() == 2


def test_frozen_perceptual_parameters_allow_reconstruction_gradients():
    class FakeCodec(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.5))

        def forward(self, images):
            reconstruction = images * self.scale
            return {"reconstruction": reconstruction, "codebook_loss": self.scale * 0,
                    "commitment_loss": self.scale * 0 + 2}

    class FakePerceptual(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, reconstruction, image):
            return (reconstruction - image).square().mean((1, 2, 3)) * self.weight

    model = FakeCodec()
    perceptual = FakePerceptual().eval()
    run = SimpleNamespace(device="cpu", config=ExperimentConfig())
    images = torch.ones(2, 3, 4, 4)
    loss, metrics = _loss_tokenizer(model, perceptual, {"image": images}, run)
    assert loss.item() == pytest.approx(0.5 + 0.1 * 0.25 + 0.25 * 2)
    loss.backward()
    assert model.scale.grad is not None and model.scale.grad.item() != 0
    assert perceptual.weight.grad is None
    assert not perceptual.training
    assert metrics["reconstruction"] == pytest.approx(0.525)


def test_freeze_blocks_further_training(tmp_path):
    run = SimpleNamespace(path=tmp_path)
    freeze = tmp_path / "evaluation" / "analysis_001" / "freeze.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("{}")
    with pytest.raises(RuntimeError, match="forbidden"):
        _assert_mutable(run, "diffusion")


def _synthetic_cache(path: Path):
    directory = path / "data" / "tokens"
    directory.mkdir(parents=True)
    checkpoint = path / "codec.pt"
    checkpoint.write_bytes(b"a stand-in used only for checksum validation")
    codec_hash = sha256_file(checkpoint)
    atomic_json(path / "data" / "tokenizer.json", {"codec_sha256": codec_hash, "checkpoint": "codec.pt"})
    atomic_json(path / "data" / "manifest.json", {"digest": "data-v1",
                "splits": {"train": [{"id": "a", "label": 0}, {"id": "b", "label": 1}]}})
    np.save(directory / "train_tokens.npy", np.zeros((2, 16, 16), dtype=np.int64))
    np.save(directory / "train_labels.npy", np.array([0, 1], dtype=np.int64))
    atomic_json(directory / "train_ids.json", ["a", "b"])
    files = {name: sha256_file(directory / name) for name in ("train_tokens.npy", "train_labels.npy", "train_ids.json")}
    atomic_json(directory / "manifest.json", {"codec_sha256": codec_hash, "data_digest": "data-v1",
                "mean": 0.0, "std": 1.0, "identity": "fixture", "splits": {"train": {"count": 2, "files": files}}})
    return SimpleNamespace(path=path, config=ExperimentConfig())


def test_token_cache_rejects_changed_codec(tmp_path):
    run = _synthetic_cache(tmp_path)
    assert load_token_cache(run, "train")["tokens"].shape == (2, 16, 16)
    (tmp_path / "codec.pt").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="stale"):
        load_token_cache(run, "train")


def test_token_cache_rejects_changed_array_and_heldout_access(tmp_path):
    run = _synthetic_cache(tmp_path)
    with pytest.raises(ValueError, match="restricted"):
        load_token_cache(run, "test")
    (tmp_path / "data" / "tokens" / "train_tokens.npy").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="checksum"):
        load_token_cache(run, "train")


def test_latent_population_moments_use_observed_frequency_and_channels():
    vectors = torch.tensor([[1.0, 3.0], [5.0, 9.0], [100.0, 200.0]])
    frequencies = torch.tensor([3, 1, 0])
    mean, std = latent_moments(vectors, frequencies)
    observed = torch.tensor([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 5.0, 9.0], dtype=torch.float64)
    assert mean == pytest.approx(observed.mean().item())
    assert std == pytest.approx(observed.std(unbiased=False).item())
    torch.testing.assert_close(((vectors - mean) / std) * std + mean, vectors)


def test_degenerate_latent_scale_stops_without_silent_fallback():
    with pytest.raises(RuntimeError, match="degenerate"):
        latent_moments(torch.ones(3, 2), torch.tensor([1, 2, 0]))


def test_token_cache_rejects_records_with_wrong_registered_order(tmp_path):
    run = _synthetic_cache(tmp_path)
    atomic_json(tmp_path / "data" / "manifest.json", {"digest": "data-v1",
                "splits": {"train": [{"id": "b", "label": 1}, {"id": "a", "label": 0}]}})
    with pytest.raises(RuntimeError, match="registered split ordering"):
        load_token_cache(run, "train")


def test_tokenizer_cannot_freeze_before_configured_training_endpoint(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from src import tokenization

    config = ExperimentConfig()
    config = replace(config, tokenizer=replace(config.tokenizer,
        training=replace(config.tokenizer.training, epochs=100)))
    run = SimpleNamespace(path=tmp_path, config=config,
                          lock=nullcontext, verify=lambda: None)
    monkeypatch.setattr(tokenization, "prepare_data", lambda run: {})
    monkeypatch.setattr(tokenization, "load_model_checkpoint",
                        lambda run, family, kind: {"epoch": 25})
    with pytest.raises(RuntimeError, match="every configured tokenizer epoch"):
        tokenization.freeze_tokenizer(run)
