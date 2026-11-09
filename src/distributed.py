"""Sample-exact global batches and FP32 distributed gradient accumulation.

The notebook launcher owns process lifetimes. This module never starts workers,
selects devices, downloads data, or initializes a process group on import.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, default_collate
from tqdm.auto import tqdm

from .config import TrainingConfig


def global_batches(count: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """A private permutation; never pad the last global batch or drop examples."""
    if count < 1 or batch_size < 1:
        raise ValueError("Training needs a nonempty dataset and positive global batch size.")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    for start in range(0, count, batch_size):
        yield order[start:start + batch_size]


def rank_batches(count: int, batch_size: int, seed: int, rank: int,
                 world_size: int) -> Iterator[tuple[list[int], int]]:
    """Shard each global batch, retaining its actual denominator on every rank."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank or world size.")
    for indices in global_batches(count, batch_size, seed):
        yield indices[rank::world_size], len(indices)


def _zero_sample(value: Any) -> Any:
    """An empty rank contributes a synthetic zero, never a duplicated example."""
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, dict):
        return {key: _zero_sample(item) for key, item in value.items()}
    if isinstance(value, (int, float)):
        return type(value)(0)
    if isinstance(value, str):
        return "__empty_rank__"
    raise TypeError(f"Unsupported training sample value: {type(value).__name__}")


def run_distributed_epoch(model: DistributedDataParallel, dataset: Dataset,
                          config: TrainingConfig, epoch: int, loss_function: Callable,
                          optimizer: torch.optim.Optimizer,
                          description: str = "tokenizer train") -> tuple[dict[str, float], int]:
    """Match a global mean loss even with unequal shards and short microbatches.

    DDP averages gradients by world size, so every local microbatch mean is
    multiplied by ``world_size * local_count / actual_global_count``. Only the
    final local backward synchronizes; an empty shard executes one zero-weight
    synthetic forward/backward to participate in that same collective.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model.train()
    totals: dict[str, float] = {}
    seen = updates = 0
    batches = rank_batches(len(dataset), config.effective_batch,
                           config.seed + 10_000 + epoch, rank, world_size)
    with torch.enable_grad():
        for indices, global_count in tqdm(batches, desc=description, leave=False,
                                          total=(len(dataset) + config.effective_batch - 1) // config.effective_batch,
                                          disable=rank != 0):
            optimizer.zero_grad(set_to_none=True)
            portions = [indices[start:start + config.microbatch]
                        for start in range(0, len(indices), config.microbatch)] or [[]]
            for number, portion in enumerate(portions):
                samples = [dataset[index] for index in portion] if portion else [_zero_sample(dataset[0])]
                batch = default_collate(samples)
                sync_context = nullcontext() if number == len(portions) - 1 else model.no_sync()
                # no_sync must enclose the forward as well as the backward.
                with sync_context:
                    loss, metrics = loss_function(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss on rank {rank} in {description}.")
                    (loss * (world_size * len(portion) / global_count)).backward()
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * len(portion)
                seen += len(portion)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
    # Float64 metric arithmetic does not change FP32 model/gradient arithmetic.
    names = sorted(totals)
    device = next(model.parameters()).device
    values = torch.tensor([totals[name] for name in names] + [seen], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    combined = values.cpu().tolist()
    if int(combined[-1]) != len(dataset):
        raise RuntimeError("Distributed sample accounting disagrees with the training manifest.")
    return {name: value / combined[-1] for name, value in zip(names, combined[:-1], strict=True)}, updates


def gather_rank_states(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Collect portable RNG states once per completed epoch, only on rank zero."""
    states: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(state, states, dst=0)
    return states


def restore_rank_state(checkpoint: dict[str, Any], topology: dict[str, Any], rank: int,
                       restore: Callable[[dict[str, Any]], None]) -> None:
    """Reject resuming under another topology before consuming any rank's RNG."""
    if checkpoint.get("distributed", {}).get("topology") != topology:
        raise RuntimeError("Distributed checkpoint topology changed; create a fresh run.")
    states = checkpoint["distributed"].get("rng_by_rank", [])
    if len(states) != topology["world_size"] or not 0 <= rank < len(states):
        raise RuntimeError("Distributed checkpoint is missing per-rank RNG states.")
    restore(states[rank])
