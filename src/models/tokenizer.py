"""Unconditional factor-eight VQ image codec; no pretrained generator weights."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import TokenizerConfig


def _groups(channels: int) -> int:
    return next(group for group in range(min(32, channels), 0, -1) if channels % group == 0)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class VQAutoencoder(nn.Module):
    """A frozen instance defines both priors' code vocabulary and image decoder.

    ``forward`` returns raw, unweighted codebook/commitment losses. The caller
    adds ``config.commitment_weight * commitment_loss`` to its reconstruction
    and codebook losses. Reconstruction gradients reach the encoder through
    the straight-through estimator, but do not update the embedding dictionary.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        if len(config.channels) != 3 or min(config.channels) < 1:
            raise ValueError("The image codec requires three positive channel widths.")
        if config.residual_blocks < 1 or min(config.codebook_size, config.embedding_dim) < 1:
            raise ValueError("Invalid codec depth or codebook shape.")
        self.config = config
        self.image_size = 128
        self.latent_size = 16
        encoder: list[nn.Module] = []
        previous = 3
        for width in config.channels:
            encoder.append(nn.Conv2d(previous, width, 4, stride=2, padding=1))
            encoder.extend(_ResidualBlock(width) for _ in range(config.residual_blocks))
            previous = width
        encoder.extend((nn.GroupNorm(_groups(previous), previous), nn.SiLU(),
                        nn.Conv2d(previous, config.embedding_dim, 1)))
        self.encoder = nn.Sequential(*encoder)
        self.codebook = nn.Embedding(config.codebook_size, config.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / config.codebook_size,
                         1.0 / config.codebook_size)
        decoder: list[nn.Module] = [nn.Conv2d(config.embedding_dim, previous, 3, padding=1)]
        widths = tuple(reversed(config.channels))
        for index, width in enumerate(widths):
            decoder.extend(_ResidualBlock(width) for _ in range(config.residual_blocks))
            next_width = widths[index + 1] if index + 1 < len(widths) else width
            decoder.extend((nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(width, next_width, 3, padding=1)))
        decoder.extend((nn.GroupNorm(_groups(widths[-1]), widths[-1]), nn.SiLU(),
                        nn.Conv2d(widths[-1], 3, 3, padding=1), nn.Tanh()))
        self.decoder = nn.Sequential(*decoder)

    def _validate_images(self, images: Tensor) -> None:
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected images shaped [B, 3, 128, 128].")
        if not images.is_floating_point() or images.shape[0] == 0:
            raise ValueError("Images must be a nonempty floating-point batch.")

    @torch.no_grad()
    def nearest_codes(self, embeddings: Tensor, chunk_size: int = 4096) -> Tensor:
        """Exact dictionary search, chunked over positions; ties use lowest ID.

        Inputs are in the ORIGINAL codec embedding units, not normalized
        diffusion units. No approximate search, clipping or codebook updates.
        """
        if embeddings.ndim != 4 or embeddings.shape[1] != self.config.embedding_dim:
            raise ValueError("Expected embeddings shaped [B, embedding_dim, H, W].")
        if not embeddings.is_floating_point() or embeddings.shape[0] == 0 or chunk_size < 1:
            raise ValueError("Provide a nonempty floating batch and positive chunk size.")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Cannot quantize nonfinite embeddings.")
        batch, _, height, width = embeddings.shape
        vectors = embeddings.permute(0, 2, 3, 1).reshape(-1, self.config.embedding_dim)
        dictionary = self.codebook.weight.detach()
        if vectors.device != dictionary.device:
            raise ValueError("Embeddings and codec must be on the same device.")
        dictionary = dictionary.to(dtype=torch.float32)
        dictionary_norm = dictionary.square().sum(dim=1).unsqueeze(0)
        pieces: list[Tensor] = []
        for start in range(0, vectors.shape[0], chunk_size):
            values = vectors[start:start + chunk_size].float()
            distances = values.square().sum(dim=1, keepdim=True) + dictionary_norm
            distances = distances - 2.0 * (values @ dictionary.t())
            pieces.append(distances.argmin(dim=1))
        return torch.cat(pieces).reshape(batch, height, width)

    def lookup(self, ids: Tensor) -> Tensor:
        if ids.ndim != 3 or ids.dtype != torch.long or ids.shape[0] == 0:
            raise ValueError("Expected nonempty integer IDs shaped [B, H, W].")
        if (ids < 0).any() or (ids >= self.config.codebook_size).any():
            raise ValueError("Token ID is outside the codec vocabulary.")
        return self.codebook(ids).permute(0, 3, 1, 2).contiguous()

    def encode(self, images: Tensor) -> Tensor:
        self._validate_images(images)
        return self.nearest_codes(self.encoder(images))

    def decode(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 4 or embeddings.shape[1:] != (
            self.config.embedding_dim, self.latent_size, self.latent_size
        ):
            raise ValueError("Expected codec embeddings shaped [B, embedding_dim, 16, 16].")
        return self.decoder(embeddings)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        self._validate_images(images)
        encoded = self.encoder(images)
        ids = self.nearest_codes(encoded)
        quantized = self.lookup(ids)
        straight_through = encoded + (quantized - encoded).detach()
        return {
            "reconstruction": self.decode(straight_through),
            "tokens": ids,
            "codebook_loss": F.mse_loss(quantized, encoded.detach()),
            "commitment_loss": F.mse_loss(encoded, quantized.detach()),
        }
