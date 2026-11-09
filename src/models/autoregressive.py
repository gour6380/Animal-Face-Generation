"""Class-prefix causal Transformer with an explicit, non-square-safe KV cache."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import AutoregressiveConfig


@dataclass
class _StaticLayerCache:
    """Private inference buffer; its unused suffix is never passed to attention."""
    key: Tensor
    value: Tensor
    length: int = 0


LayerCache = tuple[Tensor, Tensor] | _StaticLayerCache
AttentionCache = list[LayerCache]


class _Attention(nn.Module):
    def __init__(self, config: AutoregressiveConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, cache: LayerCache | None) -> tuple[Tensor, LayerCache]:
        batch, length, width = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        def split(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = split(query), split(key), split(value)
        past_length = 0
        if isinstance(cache, _StaticLayerCache):
            past_length = cache.length
            end = past_length + length
            cache.key[:, :, past_length:end].copy_(key)
            cache.value[:, :, past_length:end].copy_(value)
            cache.length = end
            key, value = cache.key[:, :, :end], cache.value[:, :, :end]
        elif cache is not None:
            old_key, old_value = cache
            past_length = old_key.shape[2]
            key, value = torch.cat((old_key, key), dim=2), torch.cat((old_value, value), dim=2)
        # One new token may attend to the whole populated prefix. Avoid building
        # an all-True mask at every decoding step. Multi-token cached suffixes
        # still need an offset causal mask (SDPA's square causal flag is wrong).
        allowed = None
        causal = past_length == 0 and length > 1
        if past_length and length > 1:
            query_positions = past_length + torch.arange(length, device=inputs.device)
            key_positions = torch.arange(key.shape[2], device=inputs.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        updated = cache if isinstance(cache, _StaticLayerCache) else (key, value)
        return self.output_dropout(self.projection(attended)), updated


class _Block(nn.Module):
    def __init__(self, config: AutoregressiveConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = _Attention(config)
        self.feedforward_norm = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(nn.Linear(config.width, config.feedforward_dim),
                                          nn.GELU(), nn.Linear(config.feedforward_dim, config.width),
                                          nn.Dropout(config.dropout))

    def forward(self, inputs: Tensor, cache: LayerCache | None) -> tuple[Tensor, LayerCache]:
        attention, new_cache = self.attention(self.attention_norm(inputs), cache)
        hidden = inputs + attention
        return hidden + self.feedforward(self.feedforward_norm(hidden)), new_cache


class ARTransformer(nn.Module):
    """Class IDs K..K+C-1 are input prefixes; outputs contain only image IDs.

    Token position zero receives the class prefix and predicts image token zero.
    The remaining positions receive preceding image tokens in row-major order.
    Learned input embeddings and the output projection are NOT weight tied.
    """

    def __init__(self, config: AutoregressiveConfig, codebook_size: int = 512,
                 sequence_length: int = 256, num_classes: int = 3) -> None:
        super().__init__()
        if config.heads < 1 or config.width % config.heads or config.layers < 1:
            raise ValueError("Invalid Transformer width, attention heads or depth.")
        if not 0 <= config.dropout < 1 or min(codebook_size, sequence_length, num_classes) < 1:
            raise ValueError("Invalid vocabulary, sequence length or dropout.")
        latent_size = math.isqrt(sequence_length)
        if latent_size * latent_size != sequence_length:
            raise ValueError("The image-token sequence must form a square grid.")
        self.config = config
        self.codebook_size, self.sequence_length, self.num_classes = codebook_size, sequence_length, num_classes
        self.latent_size = latent_size
        self.token_embedding = nn.Embedding(codebook_size + num_classes, config.width)
        self.position_embedding = nn.Embedding(sequence_length, config.width)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, codebook_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, input_ids: Tensor, cache: AttentionCache | None,
                 *, validate_values: bool = True) -> tuple[Tensor, AttentionCache]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError("Expected nonempty input IDs shaped [B, T] with dtype torch.long.")
        if validate_values and ((input_ids < 0).any() or (input_ids >= self.codebook_size + self.num_classes).any()):
            raise ValueError("Input token is outside the input vocabulary.")
        past_length = 0
        if cache is not None:
            if len(cache) != len(self.blocks):
                raise ValueError("KV cache must contain one key/value pair per block.")
            first = cache[0]
            past_length = first.length if isinstance(first, _StaticLayerCache) else first[0].shape[2]
            for layer in cache:
                if isinstance(layer, _StaticLayerCache):
                    key, value = layer.key, layer.value
                    if layer.length != past_length:
                        raise ValueError("Inconsistent KV cache length.")
                    capacity = self.sequence_length
                else:
                    key, value = layer
                    capacity = past_length
                expected = (input_ids.shape[0], self.config.heads, capacity,
                            self.config.width // self.config.heads)
                if key.shape != expected or value.shape != expected:
                    raise ValueError("Inconsistent KV cache shape.")
                if key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("KV cache and input IDs must share a device.")
        if past_length + input_ids.shape[1] > self.sequence_length:
            raise ValueError("Input and cached prefix exceed the configured sequence length.")
        positions = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
        hidden = self.input_dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None])
        new_cache: AttentionCache = []
        for index, block in enumerate(self.blocks):
            hidden, block_cache = block(hidden, None if cache is None else cache[index])
            new_cache.append(block_cache)
        return self.output(self.final_norm(hidden)), new_cache

    def forward(self, input_ids: Tensor) -> Tensor:
        return self._forward(input_ids, None)[0]

    def forward_cached(self, input_ids: Tensor, cache: AttentionCache | None = None) -> tuple[Tensor, AttentionCache]:
        if self.training:
            raise RuntimeError("KV-cache inference requires model.eval().")
        return self._forward(input_ids, cache)

    def _allocate_cache(self, batch_size: int) -> AttentionCache:
        shape = (batch_size, self.config.heads, self.sequence_length,
                 self.config.width // self.config.heads)
        weights = self.token_embedding.weight
        return [_StaticLayerCache(torch.empty(shape, device=weights.device, dtype=weights.dtype),
                                  torch.empty(shape, device=weights.device, dtype=weights.dtype))
                for _ in self.blocks]

    def _validate_labels(self, labels: Tensor) -> None:
        if labels.ndim != 1 or labels.dtype != torch.long or labels.numel() == 0:
            raise ValueError("Provide a nonempty vector of integer class labels.")
        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError("Class label is out of range.")

    def teacher_forcing(self, tokens: Tensor, labels: Tensor) -> Tensor:
        self._validate_labels(labels)
        if tokens.ndim not in (2, 3) or tokens.shape[0] != labels.shape[0] or tokens.dtype != torch.long:
            raise ValueError("Expected a matching batch of integer image-token sequences or grids.")
        flat = tokens.reshape(tokens.shape[0], -1)
        if flat.shape[1] != self.sequence_length or (flat < 0).any() or (flat >= self.codebook_size).any():
            raise ValueError("Image-token sequence shape or vocabulary is invalid.")
        prefix = (labels + self.codebook_size)[:, None]
        return self(torch.cat((prefix, flat[:, :-1]), dim=1))

    @torch.inference_mode()
    def generate(self, labels: Tensor, generator: torch.Generator, temperature: float = 1.0,
                 use_cache: bool = True) -> Tensor:
        self._validate_labels(labels)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        if generator.device.type != "cpu":
            raise ValueError("Sampling requires an explicit CPU generator.")
        device = self.token_embedding.weight.device
        labels = labels.to(device)
        context = (labels + self.codebook_size)[:, None]
        cache = self._allocate_cache(labels.shape[0]) if use_cache else None
        sampled: list[Tensor] = []
        was_training = self.training
        self.eval()
        try:
            for _ in range(self.sequence_length):
                if use_cache:
                    # Class IDs were checked once; later IDs come from our own
                    # codebook-only logits. Avoid accelerator scalar reads here.
                    logits, cache = self._forward(context, cache, validate_values=False)
                else:
                    logits = self._forward(context, None, validate_values=False)[0]
                probabilities = (logits[:, -1].float() / temperature).softmax(dim=-1).cpu()
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("Nonfinite autoregressive sampling probabilities.")
                next_token = torch.multinomial(probabilities, 1, generator=generator).to(device)
                sampled.append(next_token)
                context = next_token if use_cache else torch.cat((context, next_token), dim=1)
        finally:
            self.train(was_training)
        return torch.cat(sampled, dim=1).reshape(labels.shape[0], self.latent_size, self.latent_size)
