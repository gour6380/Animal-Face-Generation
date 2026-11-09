"""Class-conditioned latent U-Net, fixed-variance DDPM and deterministic DDIM."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import DiffusionConfig

SampleResult = Tensor | tuple[Tensor, list[Tensor]]


def ddim_timesteps(total_timesteps: int, steps: int) -> list[int]:
    """The exact descending, rounded linspace used in sampling and saved protocols."""
    if not 2 <= steps <= total_timesteps:
        raise ValueError("DDIM steps must be between two and the training timestep count.")
    return torch.linspace(0, total_timesteps - 1, steps, device="cpu", dtype=torch.float32).round().long().flip(0).tolist()


def _groups(channels: int) -> int:
    return next(group for group in range(min(32, channels), 0, -1) if channels % group == 0)


class _ConditionedResidual(nn.Module):
    def __init__(self, inputs: int, outputs: int, condition_width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(inputs), inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.condition = nn.Linear(condition_width, 2 * outputs)
        self.norm2 = nn.GroupNorm(_groups(outputs), outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(inputs) + self.conv2(F.silu(hidden))


class _SpatialAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Attention channel width must be divisible by heads.")
        self.heads = heads
        self.norm = nn.GroupNorm(_groups(width), width)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.output = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, width, height, columns = inputs.shape
        query, key, value = self.qkv(self.norm(inputs)).chunk(3, dim=1)
        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, self.heads, width // self.heads, -1).transpose(-1, -2)
        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value))
        attended = attended.transpose(-1, -2).reshape(batch, width, height, columns)
        return inputs + self.output(attended)


class _Stage(nn.Module):
    def __init__(self, inputs: int, outputs: int, config: DiffusionConfig, size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _ConditionedResidual(inputs if i == 0 else outputs, outputs, config.time_embedding_dim)
            for i in range(config.residual_blocks)
        ])
        self.attention = (_SpatialAttention(outputs, config.attention_heads)
                          if size in config.attention_sizes else nn.Identity())

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            inputs = block(inputs, condition)
        return self.attention(inputs)


class DiffusionUNet(nn.Module):
    def __init__(self, config: DiffusionConfig, latent_channels: int = 64,
                 num_classes: int = 3, latent_size: int = 16) -> None:
        super().__init__()
        if not config.channels or min(config.channels) < 1 or config.residual_blocks < 1:
            raise ValueError("Invalid U-Net channel/depth configuration.")
        if latent_size % (2 ** (len(config.channels) - 1)):
            raise ValueError("Latent size must be divisible by all downsampling stages.")
        self.config = config
        self.latent_channels, self.latent_size, self.num_classes = latent_channels, latent_size, num_classes
        self.time_width = config.channels[0]
        self.time_mlp = nn.Sequential(nn.Linear(self.time_width, config.time_embedding_dim),
                                      nn.SiLU(), nn.Linear(config.time_embedding_dim,
                                                          config.time_embedding_dim))
        self.class_embedding = nn.Embedding(num_classes, config.time_embedding_dim)
        self.input = nn.Conv2d(latent_channels, config.channels[0], 3, padding=1)
        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, width in enumerate(config.channels):
            self.down_stages.append(_Stage(width, width, config, latent_size // (2 ** index)))
            if index + 1 < len(config.channels):
                self.downsamples.append(nn.Conv2d(width, config.channels[index + 1],
                                                   3, stride=2, padding=1))
        self.middle = _ConditionedResidual(config.channels[-1], config.channels[-1],
                                           config.time_embedding_dim)
        self.up_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(config.channels))):
            width = config.channels[index]
            self.up_stages.append(_Stage(2 * width, width, config, latent_size // (2 ** index)))
            if index > 0:
                self.upsamples.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                                      nn.Conv2d(width, config.channels[index - 1],
                                                                3, padding=1)))
        self.output = nn.Sequential(nn.GroupNorm(_groups(config.channels[0]), config.channels[0]),
                                    nn.SiLU(), nn.Conv2d(config.channels[0], latent_channels,
                                                        3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _time_embedding(self, timesteps: Tensor) -> Tensor:
        half = self.time_width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half, 1))
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
        if self.time_width % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.time_mlp(embedding)

    def _validate_inputs(self, inputs: Tensor, timesteps: Tensor, labels: Tensor) -> None:
        if inputs.ndim != 4 or inputs.shape[1:] != (
            self.latent_channels, self.latent_size, self.latent_size
        ):
            raise ValueError("Unexpected latent shape for this U-Net.")
        if timesteps.shape != (inputs.shape[0],) or labels.shape != timesteps.shape:
            raise ValueError("Provide one timestep and class label per latent.")
        if timesteps.dtype != torch.long or labels.dtype != torch.long:
            raise ValueError("Timesteps and class labels must have dtype torch.long.")
        if (timesteps < 0).any() or (timesteps >= self.config.timesteps).any():
            raise ValueError("Diffusion timestep is out of range.")
        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError("Class label is out of range.")
    def forward(self, inputs: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        self._validate_inputs(inputs, timesteps, labels)
        return self._predict_noise(inputs, timesteps, labels)

    def _predict_noise(self, inputs: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        """Internal sampler path after one boundary validation of fixed labels."""
        condition = self._time_embedding(timesteps) + self.class_embedding(labels)
        hidden = self.input(inputs)
        skips: list[Tensor] = []
        for index, stage in enumerate(self.down_stages):
            hidden = stage(hidden, condition)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.middle(hidden, condition)
        for index, stage in enumerate(self.up_stages):
            hidden = stage(torch.cat((hidden, skips.pop()), dim=1), condition)
            if index < len(self.upsamples):
                hidden = self.upsamples[index](hidden)
        return self.output(hidden)


class DiffusionSchedule(nn.Module):
    """Linear beta schedule; no clipping or intermediate codebook projection.

    Coefficients are computed on CPU in float64 then stored in float32 buffers.
    Move this module to the model device once for repeated sampling. A trajectory
    contains detached CPU snapshots of the initial state and about ten states,
    including the final state. DDIM uses eta=0 and exactly ``steps`` model calls.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        if config.timesteps < 2 or not 0 < config.beta_start < config.beta_end < 1:
            raise ValueError("Invalid diffusion schedule.")
        self.timesteps = config.timesteps
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps,
                               dtype=torch.float64, device="cpu")
        alphas = 1.0 - betas
        cumulative = alphas.cumprod(dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64, device="cpu"), cumulative[:-1]))
        buffers = {
            "betas": betas, "alphas": alphas, "alphas_cumprod": cumulative,
            "alphas_cumprod_previous": previous,
            "sqrt_alphas_cumprod": cumulative.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1 - cumulative).sqrt(),
            "posterior_variance": betas * (1 - previous) / (1 - cumulative),
            "posterior_mean_coef1": betas * previous.sqrt() / (1 - cumulative),
            "posterior_mean_coef2": (1 - previous) * alphas.sqrt() / (1 - cumulative),
        }
        for name, values in buffers.items():
            self.register_buffer(name, values.float())

    def _extract(self, coefficients: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
        return coefficients.to(reference.device).gather(0, timesteps).reshape(
            reference.shape[0], *([1] * (reference.ndim - 1))
        ).to(dtype=reference.dtype)

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        if noise.shape != x0.shape or timesteps.shape != (x0.shape[0],):
            raise ValueError("Noise/timestep shapes do not match the clean latent batch.")
        if timesteps.dtype != torch.long or (timesteps < 0).any() or (timesteps >= self.timesteps).any():
            raise ValueError("Invalid diffusion timesteps.")
        return self._extract(self.sqrt_alphas_cumprod, timesteps, x0) * x0 + self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x0
        ) * noise

    @staticmethod
    def _record(states: list[Tensor], current: Tensor, index: int, total: int) -> None:
        if (index + 1) % max(1, total // 10) == 0 or index + 1 == total:
            states.append(current.detach().cpu().clone())

    @torch.inference_mode()
    def ddim_sample(self, model: nn.Module, initial_noise: Tensor, labels: Tensor,
                    steps: int, trajectory: bool = False) -> SampleResult:
        indices = ddim_timesteps(self.timesteps, steps)
        current = initial_noise.clone()
        states = [current.detach().cpu().clone()] if trajectory else []
        was_training = model.training
        model.eval()
        try:
            for index, timestep in enumerate(indices):
                t = torch.full((current.shape[0],), timestep, dtype=torch.long, device=current.device)
                if type(model) is DiffusionUNet:
                    if index == 0:
                        model._validate_inputs(current, t, labels)
                    # Every later t is generated by our validated schedule;
                    # repeated tensor.any() checks force accelerator synchronization.
                    epsilon = model._predict_noise(current, t, labels)
                else:
                    epsilon = model(current, t, labels)
                alpha = self._extract(self.alphas_cumprod, t, current)
                predicted_clean = (current - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
                next_timestep = indices[index + 1] if index + 1 < len(indices) else -1
                if next_timestep < 0:
                    current = predicted_clean
                else:
                    next_t = torch.full_like(t, next_timestep)
                    next_alpha = self._extract(self.alphas_cumprod, next_t, current)
                    current = next_alpha.sqrt() * predicted_clean + (1 - next_alpha).sqrt() * epsilon
                if trajectory:
                    self._record(states, current, index, len(indices))
        finally:
            model.train(was_training)
        return (current, states) if trajectory else current

    @torch.inference_mode()
    def ddpm_sample(self, model: nn.Module, initial_noise: Tensor, labels: Tensor,
                    generator: torch.Generator, trajectory: bool = False) -> SampleResult:
        if generator.device.type != "cpu":
            raise ValueError("Sampling requires an explicit CPU generator.")
        current = initial_noise.clone()
        states = [current.detach().cpu().clone()] if trajectory else []
        was_training = model.training
        model.eval()
        try:
            for index, timestep in enumerate(reversed(range(self.timesteps))):
                t = torch.full((current.shape[0],), timestep, dtype=torch.long, device=current.device)
                if type(model) is DiffusionUNet:
                    if index == 0:
                        model._validate_inputs(current, t, labels)
                    # Every later t is generated by our validated schedule;
                    # repeated tensor.any() checks force accelerator synchronization.
                    epsilon = model._predict_noise(current, t, labels)
                else:
                    epsilon = model(current, t, labels)
                alpha = self._extract(self.alphas_cumprod, t, current)
                predicted_clean = (current - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
                current = self._extract(self.posterior_mean_coef1, t, current) * predicted_clean + self._extract(
                    self.posterior_mean_coef2, t, current
                ) * current
                if timestep > 0:
                    noise = torch.randn(current.shape, generator=generator, device="cpu", dtype=torch.float32)
                    current = current + self._extract(self.posterior_variance, t, current).sqrt() * noise.to(
                        device=current.device, dtype=current.dtype
                    )
                if trajectory:
                    self._record(states, current, index, self.timesteps)
        finally:
            model.train(was_training)
        return (current, states) if trajectory else current
