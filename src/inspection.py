"""Lightweight architecture inspection, with no training or network activity."""
from __future__ import annotations

import torch

from .models.autoregressive import ARTransformer
from .models.diffusion import DiffusionUNet
from .models.tokenizer import VQAutoencoder
from .runs import Run


def describe_model(run: Run, family: str) -> str:
    """Construct a temporary CPU module, preserving the initialization RNG stream."""
    run.verify()
    cfg = run.config
    with torch.random.fork_rng(devices=[]):
        if family == "tokenizer":
            model = VQAutoencoder(cfg.tokenizer)
        elif family == "diffusion":
            model = DiffusionUNet(cfg.diffusion, latent_channels=cfg.tokenizer.embedding_dim,
                                  num_classes=3, latent_size=cfg.latent_size)
        elif family == "autoregressive":
            model = ARTransformer(cfg.autoregressive, codebook_size=cfg.tokenizer.codebook_size,
                                  sequence_length=cfg.latent_size ** 2, num_classes=3)
        else:
            raise ValueError("Unknown model family.")
        count = sum(p.numel() for p in model.parameters())
        return f"{family} — {count:,} parameters\n\n{model}"
