"""Authored model-invariant tests. They have not been executed at handoff."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from src.config import AutoregressiveConfig, DiffusionConfig, TokenizerConfig
from src.models.autoregressive import ARTransformer
from src.models.diffusion import DiffusionSchedule, DiffusionUNet
from src.models.tokenizer import VQAutoencoder


def small_codec() -> VQAutoencoder:
    return VQAutoencoder(TokenizerConfig(
        channels=(8, 16, 16), residual_blocks=1, codebook_size=8, embedding_dim=4,
    ))


def small_transformer() -> ARTransformer:
    return ARTransformer(AutoregressiveConfig(
        layers=2, width=16, heads=2, feedforward_dim=32, dropout=0.0,
    ), codebook_size=8, sequence_length=4).eval()


def test_codec_shapes_and_reconstruction_range() -> None:
    model = small_codec()
    images = torch.zeros(1, 3, 128, 128)
    output = model(images)
    assert output["tokens"].shape == (1, 16, 16)
    assert output["tokens"].dtype == torch.long
    assert output["reconstruction"].shape == images.shape
    assert output["reconstruction"].abs().max() <= 1
    assert model.lookup(output["tokens"]).shape == (1, 4, 16, 16)
    assert torch.equal(model.encode(images), output["tokens"])


def test_nearest_code_lookup_chunking_and_tie_rule() -> None:
    model = small_codec()
    with torch.no_grad():
        model.codebook.weight.copy_(torch.arange(8).float()[:, None].expand(-1, 4))
        model.codebook.weight[3].copy_(model.codebook.weight[2])
    ids = torch.tensor([[[0, 2, 4, 7]]])
    embeddings = model.lookup(ids)
    assert torch.equal(model.nearest_codes(embeddings, chunk_size=1), ids)
    assert torch.equal(model.nearest_codes(embeddings, chunk_size=4096), ids)
    with pytest.raises(ValueError, match="nonfinite"):
        model.nearest_codes(torch.full((1, 4, 1, 1), float("nan")))


def test_codec_straight_through_gradient_ownership() -> None:
    model = small_codec()
    output = model(torch.randn(1, 3, 128, 128).tanh())
    output["reconstruction"].square().mean().backward()
    assert model.codebook.weight.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.decoder.parameters())


def test_codebook_and_commitment_losses_update_different_parameters() -> None:
    model = small_codec()
    images = torch.randn(1, 3, 128, 128).tanh()
    model(images)["codebook_loss"].backward()
    assert model.codebook.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    model(images)["commitment_loss"].backward()
    assert model.codebook.weight.grad is None
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_codec_rejects_wrong_shape_and_non_image_tokens() -> None:
    model = small_codec()
    with pytest.raises(ValueError, match="128"):
        model.encode(torch.zeros(1, 3, 64, 64))
    with pytest.raises(ValueError, match="vocabulary"):
        model.lookup(torch.tensor([[[8]]]))


def test_teacher_forcing_uses_class_prefix_and_one_position_shift() -> None:
    model = small_transformer()
    tokens = torch.tensor([[[1, 2], [3, 4]], [[4, 3], [2, 1]]])
    labels = torch.tensor([0, 2])
    expected_inputs = torch.tensor([[8, 1, 2, 3], [10, 4, 3, 2]])
    with torch.no_grad():
        actual = model.teacher_forcing(tokens, labels)
        expected = model(expected_inputs)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 4, 8)


def test_transformer_future_tokens_cannot_change_earlier_logits() -> None:
    model = small_transformer()
    original = torch.tensor([[8, 1, 2, 3]])
    changed = torch.tensor([[8, 1, 6, 7]])
    with torch.no_grad():
        first, second = model(original), model(changed)
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_non_square_cache_mask_matches_full_prefix() -> None:
    model = small_transformer()
    inputs = torch.tensor([[8, 1, 2, 3], [10, 4, 3, 2]])
    with torch.no_grad():
        expected = model(inputs)
        first, cache = model.forward_cached(inputs[:, :2])
        second, cache = model.forward_cached(inputs[:, 2:3], cache)
        third, cache = model.forward_cached(inputs[:, 3:], cache)
    torch.testing.assert_close(torch.cat((first, second, third), dim=1), expected,
                               rtol=1e-5, atol=1e-6)
    assert all(key.shape == (2, 2, 4, 8) and value.shape == key.shape
               for key, value in cache)


def test_multi_token_cached_suffix_is_causal() -> None:
    model = small_transformer()
    inputs = torch.tensor([[8, 1, 2, 3]])
    with torch.no_grad():
        _, cache = model.forward_cached(inputs[:, :1])
        suffix, _ = model.forward_cached(inputs[:, 1:], cache)
        expected = model(inputs)[:, 1:]
    torch.testing.assert_close(suffix, expected, rtol=1e-5, atol=1e-6)


def test_cached_generation_matches_uncached_and_excludes_class_tokens() -> None:
    model = small_transformer()
    labels = torch.tensor([0, 1, 2])
    cached = model.generate(labels, torch.Generator(device="cpu").manual_seed(5))
    uncached = model.generate(labels, torch.Generator(device="cpu").manual_seed(5), use_cache=False)
    assert torch.equal(cached, uncached)
    assert cached.shape == (3, 2, 2)
    assert cached.min() >= 0 and cached.max() < 8


def test_cache_rejects_training_mode_and_overlong_input() -> None:
    model = small_transformer().train()
    with pytest.raises(RuntimeError, match="eval"):
        model.forward_cached(torch.tensor([[8]]))
    model.eval()
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.tensor([[8, 1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="Temperature"):
        model.generate(torch.tensor([0]), torch.Generator(), temperature=0)


def test_q_sample_matches_closed_form_and_validates_timesteps() -> None:
    schedule = DiffusionSchedule(replace(DiffusionConfig(), timesteps=8))
    x0 = torch.ones(2, 4, 2, 2)
    noise = torch.full_like(x0, 2)
    timesteps = torch.tensor([0, 7])
    alpha = schedule.alphas_cumprod[timesteps][:, None, None, None]
    expected = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
    torch.testing.assert_close(schedule.q_sample(x0, timesteps, noise), expected)
    assert all(buffer.device.type == "cpu" and buffer.dtype == torch.float32
               for buffer in schedule.buffers())
    assert schedule.posterior_variance[0] == 0
    with pytest.raises(ValueError, match="timesteps"):
        schedule.q_sample(x0, torch.tensor([-1, 8]), noise)


class ZeroCleanTarget(nn.Module):
    """Oracle epsilon predictor for an identically zero clean latent."""
    def __init__(self, schedule: DiffusionSchedule) -> None:
        super().__init__()
        self.schedule = schedule
        self.calls: list[int] = []

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(timesteps[0]))
        variance = 1 - self.schedule.alphas_cumprod[timesteps]
        return inputs / variance[:, None, None, None].sqrt()


def test_ddim_oracle_endpoint_exact_call_count_and_trajectory() -> None:
    schedule = DiffusionSchedule(replace(DiffusionConfig(), timesteps=8))
    model = ZeroCleanTarget(schedule)
    initial = torch.randn(2, 4, 2, 2)
    output, trajectory = schedule.ddim_sample(model, initial, torch.tensor([0, 1]),
                                              steps=4, trajectory=True)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=0)
    assert len(model.calls) == 4
    assert model.calls[0] == 7 and model.calls[-1] == 0
    assert model.calls == sorted(set(model.calls), reverse=True)
    assert model.training
    torch.testing.assert_close(trajectory[0], initial)
    torch.testing.assert_close(trajectory[-1], output)


def test_ddpm_zero_endpoint_and_no_noise_draw_at_last_step() -> None:
    schedule = DiffusionSchedule(replace(DiffusionConfig(), timesteps=8))
    model = ZeroCleanTarget(schedule)
    initial = torch.randn(2, 4, 2, 2)
    generator = torch.Generator().manual_seed(71)
    expected_generator = torch.Generator().manual_seed(71)
    output = schedule.ddpm_sample(model, initial, torch.tensor([0, 1]), generator)
    for _ in range(7):
        torch.randn(initial.shape, generator=expected_generator)
    assert torch.equal(generator.get_state(), expected_generator.get_state())
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-5, rtol=0)
    assert model.calls == list(reversed(range(8)))


def test_small_unet_preserves_latent_shape_and_initializes_finitely() -> None:
    config = replace(DiffusionConfig(), channels=(8, 16), residual_blocks=1,
                     attention_heads=2, attention_sizes=(2,), time_embedding_dim=16,
                     timesteps=8)
    model = DiffusionUNet(config, latent_channels=4, latent_size=4)
    inputs = torch.randn(2, 4, 4, 4)
    output = model(inputs, torch.tensor([0, 7]), torch.tensor([0, 2]))
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.output[-1].weight.grad is not None


def test_invalid_model_shapes_fail_before_training() -> None:
    with pytest.raises(ValueError, match="width"):
        ARTransformer(AutoregressiveConfig(width=17, heads=2))
    with pytest.raises(ValueError, match="square"):
        ARTransformer(AutoregressiveConfig(), sequence_length=15)
    with pytest.raises(ValueError, match="schedule"):
        DiffusionSchedule(replace(DiffusionConfig(), timesteps=1))
