"""Offline numerical checks for bounded inference and metric optimizations."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.config import AutoregressiveConfig, DiffusionConfig
from src.evaluate import kernel_inception_distance
from src.io import read_json, sha256_file
from src.models.autoregressive import ARTransformer
from src.models.diffusion import DiffusionSchedule, DiffusionUNet
from src.sample import _atomic_png, generate_samples, setting_spec


def _transformer():
    return ARTransformer(AutoregressiveConfig(layers=2, width=16, heads=2,
                         feedforward_dim=32, dropout=0), codebook_size=8, sequence_length=16).eval()


def test_static_cache_matches_dynamic_chunks_without_reallocating_storage():
    model = _transformer()
    ids = torch.tensor([[8, 1, 2, 3, 4, 5, 6], [9, 2, 1, 0, 7, 6, 5]])
    with torch.inference_mode():
        static = model._allocate_cache(2)
        pointers = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in static]
        expected = model(ids)
        outputs = []
        for start, end in ((0, 2), (2, 3), (3, 7)):
            logits, static = model.forward_cached(ids[:, start:end], static)
            outputs.append(logits)
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)
        assert [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in static] == pointers
        assert all(layer.length == 7 for layer in static)
        with pytest.raises(ValueError, match="vocabulary"):
            model.forward_cached(torch.tensor([[11], [11]]), static)
        with pytest.raises(ValueError, match="sequence length"):
            model.forward_cached(torch.zeros(2, 10, dtype=torch.long), static)


@pytest.mark.parametrize("temperature", [0.8, 1.0, 1.2])
def test_static_generation_matches_dynamic_sampling_and_rng(temperature):
    model = _transformer()
    labels = torch.tensor([0, 1, 2])
    actual_rng = torch.Generator().manual_seed(971)
    reference_rng = torch.Generator().manual_seed(971)
    actual = model.generate(labels, actual_rng, temperature=temperature)
    with torch.inference_mode():
        context, cache, tokens = (labels + 8)[:, None], None, []
        for _ in range(16):
            logits, cache = model.forward_cached(context, cache)
            probabilities = (logits[:, -1].float() / temperature).softmax(-1).cpu()
            context = torch.multinomial(probabilities, 1, generator=reference_rng)
            tokens.append(context)
    assert torch.equal(actual.flatten(1), torch.cat(tokens, dim=1))
    assert torch.equal(actual_rng.get_state(), reference_rng.get_state())


class _PublicForward(nn.Module):
    """Force the sampler's generic public-forward path as an independent baseline."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, timesteps, labels):
        return self.model(inputs, timesteps, labels)


@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_diffusion_validates_once_and_matches_public_forward(monkeypatch, sampler):
    config = replace(DiffusionConfig(), channels=(8, 16), residual_blocks=1,
                     attention_heads=2, attention_sizes=(2,), time_embedding_dim=16, timesteps=8)
    model = DiffusionUNet(config, latent_channels=4, latent_size=4).eval()
    # Exercise nonzero noise predictions, not just the zero-initialized head.
    with torch.no_grad():
        model.output[-1].weight.normal_(std=0.03)
    schedule = DiffusionSchedule(config)
    noise, labels = torch.randn(2, 4, 4, 4), torch.tensor([0, 2])
    validate = model._validate_inputs
    calls = []

    def tracked(*args):
        calls.append(1)
        return validate(*args)

    monkeypatch.setattr(model, "_validate_inputs", tracked)
    one, two = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    if sampler == "ddim":
        actual = schedule.ddim_sample(model, noise, labels, steps=4)
        assert len(calls) == 1
        expected = schedule.ddim_sample(_PublicForward(model), noise, labels, steps=4)
    else:
        actual = schedule.ddpm_sample(model, noise, labels, one)
        assert len(calls) == 1
        expected = schedule.ddpm_sample(_PublicForward(model), noise, labels, two)
        assert torch.equal(one.get_state(), two.get_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="label"):
        schedule.ddim_sample(model, noise, torch.tensor([0, 3]), steps=4)


def _direct_kid(real, fake, subset_size, subsets, seed):
    rng, values = np.random.default_rng(seed), []
    for _ in range(subsets):
        x, y = real[rng.choice(len(real), subset_size, replace=False)], fake[rng.choice(len(fake), subset_size, replace=False)]
        xx, yy, xy = (x @ x.T / x.shape[1] + 1) ** 3, (y @ y.T / y.shape[1] + 1) ** 3, (x @ y.T / x.shape[1] + 1) ** 3
        values.append(((xx.sum() - np.trace(xx)) + (yy.sum() - np.trace(yy))) /
                      (subset_size * (subset_size - 1)) - 2 * xy.mean())
    return np.mean(values), np.std(values)


@pytest.mark.parametrize("subsets", [2, 100])
def test_reused_kernels_match_original_subset_draws_and_statistics(subsets):
    rng = np.random.default_rng(51)
    real, fake = rng.normal(size=(70, 128)), rng.normal(size=(60, 128))
    expected = _direct_kid(real, fake, subset_size=25, subsets=subsets, seed=81)
    actual = kernel_inception_distance(real, fake, subset_size=25, subsets=subsets, seed=81)
    assert actual["mean"] == pytest.approx(expected[0], abs=1e-12)
    assert actual["subset_std"] == pytest.approx(expected[1], abs=1e-12)


def test_atomic_png_hashes_encoded_bytes_and_recovery_rejects_changed_image(tmp_path):
    path = tmp_path / "class" / "sample.png"
    pixels = torch.randint(256, (3, 8, 8), dtype=torch.uint8)
    checksum = _atomic_png(path, pixels)
    assert checksum == sha256_file(path)
    assert _atomic_png(path, pixels) == checksum
    with Image.open(path) as image:
        assert np.array_equal(np.array(image), pixels.permute(1, 2, 0).numpy())
    with pytest.raises(RuntimeError, match="Existing sample differs"):
        _atomic_png(path, pixels ^ 255)


def test_bounded_png_pipeline_commits_ordered_balanced_manifest_and_reuses(tmp_path, monkeypatch):
    from src import evaluate, sample
    classes = ("cat", "dog", "wild")
    spec = setting_spec("diffusion", "ddim_50")
    freeze = {"settings": {"ddim_50": spec}, "evaluation_config": {"sampling_seed": 71, "generation_batch": 2}}
    run = SimpleNamespace(path=tmp_path, config=SimpleNamespace(data=SimpleNamespace(classes=classes)), lock=nullcontext)
    monkeypatch.setattr(evaluate, "load_freeze", lambda *args: freeze)
    monkeypatch.setattr(evaluate, "_prepare_reference", lambda *args: {"n_per_class": 3})
    calls = []

    class Wrapper:
        def generate(self, class_name, count, seed, **kwargs):
            calls.append((class_name, count, seed))
            return torch.randint(256, (count, 3, 8, 8), dtype=torch.uint8,
                                 generator=torch.Generator().manual_seed(seed))

    monkeypatch.setattr(sample, "load_generator", lambda *args: Wrapper())
    result = generate_samples(run, "diffusion", "ddim_50")
    assert result["complete"]
    assert [(r["label"], r["index"]) for r in result["records"]] == [(label, index) for label in range(3) for index in range(3)]
    assert len(calls) == 6
    assert generate_samples(run, "diffusion", "ddim_50") == result
    assert len(calls) == 6
    directory = tmp_path / "evaluation/analysis_001/samples/ddim_50"
    assert read_json(directory / "manifest.json") == result
    (directory / result["records"][0]["file"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Sample changed"):
        generate_samples(run, "diffusion", "ddim_50")


def test_png_worker_failure_never_marks_manifest_complete(tmp_path, monkeypatch):
    from src import evaluate, sample
    freeze = {"settings": {"ddim_50": setting_spec("diffusion", "ddim_50")},
              "evaluation_config": {"sampling_seed": 71, "generation_batch": 2}}
    run = SimpleNamespace(path=tmp_path, config=SimpleNamespace(data=SimpleNamespace(classes=("cat",))), lock=nullcontext)
    monkeypatch.setattr(evaluate, "load_freeze", lambda *args: freeze)
    monkeypatch.setattr(evaluate, "_prepare_reference", lambda *args: {"n_per_class": 2})
    monkeypatch.setattr(sample, "load_generator", lambda *args: SimpleNamespace(generate=lambda *args, **kwargs: torch.zeros(2, 3, 8, 8, dtype=torch.uint8)))

    def fail(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(sample, "_atomic_png", fail)
    with pytest.raises(OSError, match="full disk"):
        generate_samples(run, "diffusion", "ddim_50")
    manifest = read_json(tmp_path / "evaluation/analysis_001/samples/ddim_50/manifest.json")
    assert not manifest["complete"]


def test_generated_features_decode_the_same_bytes_that_were_verified(tmp_path):
    from src.evaluate import _GeneratedImages
    pixels = torch.randint(256, (3, 8, 8), dtype=torch.uint8)
    checksum = _atomic_png(tmp_path / "sample.png", pixels)
    dataset = _GeneratedImages(tmp_path, [{"file": "sample.png", "sha256": checksum}])
    assert torch.equal(dataset[0], pixels)
    (tmp_path / "sample.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Generated image changed"):
        dataset[0]
