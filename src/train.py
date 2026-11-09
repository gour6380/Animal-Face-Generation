"""Notebook-driven training with epoch transactions and sample-weighted accumulation."""
from __future__ import annotations

import gc
import math
import os
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import nullcontext, suppress
from datetime import timedelta
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .checkpoints import (
    capture_rng,
    load_training_state,
    read_training_state,
    restore_rng,
    save_architecture,
    save_epoch,
)
from .config import TrainingConfig
from .data import ImageDataset
from .data import load_prepared_data as prepare_data
from .device import memory_snapshot, seed_everything, synchronize
from .distributed import (
    gather_rank_states,
    global_batches,
    restore_rank_state,
    run_distributed_epoch,
)
from .models.autoregressive import ARTransformer
from .models.diffusion import DiffusionSchedule, DiffusionUNet
from .models.tokenizer import VQAutoencoder
from .tokenization import load_frozen_tokenizer, load_token_cache
from .tracking import TrainingTracker
from .weights import load_lpips


def microbatches(batch: dict[str, Any], size: int) -> Iterator[dict[str, Any]]:
    """Partition one effective batch; retain its short final microbatch."""
    count = len(batch["label"])
    for start in range(0, count, size):
        yield {key: value[start:start + size] for key, value in batch.items()}


def lr_multiplier(epoch_index: int, config: TrainingConfig) -> float:
    """Zero-based epoch learning-rate multiplier, with an exact final floor."""
    if config.warmup_epochs == 0:
        return 1.0
    if epoch_index < config.warmup_epochs:
        return (epoch_index + 1) / config.warmup_epochs
    progress = min(1.0, (epoch_index - config.warmup_epochs)
                   / max(1, config.epochs - config.warmup_epochs - 1))
    return config.minimum_lr_fraction + (1 - config.minimum_lr_fraction) * (1 + math.cos(math.pi * progress)) / 2


def update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    """EMA includes floating parameters/buffers; integer buffers copy exactly."""
    with torch.no_grad():
        for name, current in model.state_dict().items():
            if current.is_floating_point():
                ema[name].lerp_(current.detach(), 1 - decay)
            else:
                ema[name].copy_(current)


class TokenDataset(Dataset):
    def __init__(self, cache: dict[str, Any]) -> None:
        self.tokens, self.labels, self.ids = cache["tokens"], cache["labels"], cache["ids"]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"tokens": self.tokens[index], "label": self.labels[index], "id": self.ids[index]}


def _loader(dataset: Dataset, config: TrainingConfig, epoch: int, training: bool) -> DataLoader:
    # Private generator separates data order from dropout, diffusion noise and model init.
    generator = torch.Generator().manual_seed(config.seed + 10_000 + (epoch if training else 0))
    if training:
        return DataLoader(dataset, batch_sampler=list(global_batches(
            len(dataset), config.effective_batch, config.seed + 10_000 + epoch)),
            num_workers=0, generator=generator)
    return DataLoader(dataset, batch_size=config.microbatch, shuffle=False,
                      drop_last=False, num_workers=0, generator=generator)


def _assert_mutable(run: Any, family: str) -> None:
    if list((run.path / "evaluation").glob("analysis_*/freeze.json")):
        raise RuntimeError("Training is forbidden after final evaluation is frozen. Create a fresh run.")
    if family == "tokenizer" and (run.path / "data" / "tokenizer.json").exists():
        raise RuntimeError("The shared tokenizer is frozen and cannot be retrained in this run.")


def _loss_tokenizer(model: nn.Module, perceptual: nn.Module, batch: dict[str, Any],
                    run: Any) -> tuple[torch.Tensor, dict[str, float]]:
    image = batch["image"].to(run.device)
    output = model(image)
    reconstruction = output["reconstruction"]
    l1 = F.l1_loss(reconstruction, image)
    # Frozen LPIPS parameters must not disable gradients with respect to reconstruction.
    perceptual_loss = perceptual(reconstruction, image).mean()
    reconstruction_loss = l1 + run.config.tokenizer.perceptual_weight * perceptual_loss
    loss = (reconstruction_loss + output["codebook_loss"]
            + run.config.tokenizer.commitment_weight * output["commitment_loss"])
    return loss, {"loss": loss.detach().item(), "reconstruction": reconstruction_loss.detach().item(),
                  "l1": l1.detach().item(), "lpips": perceptual_loss.detach().item(),
                  "codebook": output["codebook_loss"].detach().item(),
                  "commitment": output["commitment_loss"].detach().item()}


def _noise(shape: torch.Size, generator: torch.Generator, device: str | torch.device) -> torch.Tensor:
    # CPU generators are portable across MPS/CUDA/CPU and deliberately independent.
    return torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu").to(device)


def _loss_diffusion(model: nn.Module, codec: VQAutoencoder, schedule: DiffusionSchedule,
                    cache: dict[str, Any], batch: dict[str, Any], run: Any,
                    generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        latent = codec.lookup(batch["tokens"].to(run.device))
        latent = (latent - cache["mean"]) / cache["std"]
        timestep = torch.randint(run.config.diffusion.timesteps, (len(latent),), generator=generator).to(run.device)
        noise = _noise(latent.shape, generator, run.device)
        noisy = schedule.q_sample(latent, timestep, noise)
    prediction = model(noisy, timestep, batch["label"].to(run.device))
    loss = F.mse_loss(prediction, noise)
    return loss, {"loss": loss.detach().item(), "noise_mse": loss.detach().item()}


def _loss_autoregressive(model: ARTransformer, batch: dict[str, Any], run: Any) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = batch["tokens"].flatten(1).to(run.device)
    logits = model.teacher_forcing(tokens, batch["label"].to(run.device))
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1))
    return loss, {"loss": loss.detach().item(), "cross_entropy": loss.detach().item()}


def _run_epoch(model: nn.Module, loader: DataLoader, config: TrainingConfig,
               loss_function: Callable, optimizer: torch.optim.Optimizer | None,
               ema: dict[str, torch.Tensor] | None, ema_decay: float,
               description: str) -> tuple[dict[str, float], int]:
    """Normalize each gradient by the actual effective-batch sample count."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    seen = 0
    updates = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for effective in tqdm(loader, desc=description, leave=False):
            actual = len(effective["label"])
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in microbatches(effective, config.microbatch):
                weight = len(batch["label"])
                loss, metrics = loss_function(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {description}; configuration was not changed.")
                if training:
                    (loss * (weight / actual)).backward()
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * weight
                seen += weight
            if training:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                if ema is not None:
                    update_ema(ema, model, ema_decay)
    if not seen:
        raise ValueError("Cannot train or evaluate an empty split.")
    return {key: value / seen for key, value in totals.items()}, updates


def _validate_training_assignment(run: Any, family: str) -> None:
    """Direct calls must not silently bypass the run's recorded worker topology."""
    hardware = getattr(run, "hardware", None) or {}
    if family == "tokenizer" and hardware.get("tokenizer_world_size", 1) > 1:
        raise RuntimeError("This run requires two tokenizer workers. Use launcher.run_action('train_tokenizer', run_name).")
    if hardware.get("device") == "cuda" and hardware.get("parallel_generators") and family != "tokenizer":
        assigned = hardware["assignments"][family]
        visible = os.environ.get("AFG_WORKER_UUIDS", "").split(",")
        if os.environ.get("AFG_WORKER_ROLE") != family or visible != assigned:
            raise RuntimeError(f"Use launcher.start_training(run_name, '{family}') to honor this run's isolated GPU assignment.")


def _train(run: Any, family: str) -> dict[str, Any]:
    with run.training_lock(family):
        run.verify()
        _validate_training_assignment(run, family)
        config = getattr(run.config, family).training
        state = load_training_state(run, family)
        if state is not None and state["epoch"] >= config.epochs:
            return {"family": family, "completed_epochs": state["epoch"], "status": "reused",
                    "directory": str(run.model_dir(family))}
        _assert_mutable(run, family)
        manifest = prepare_data(run)
        seed_everything(config.seed + 101)  # model initialization stream
        torch.set_num_threads(run.config.runtime.cpu_threads)
        codec = perceptual = schedule = None
        cache: dict[str, Any] = {}
        metadata: dict[str, Any] = {"family": family, "data_digest": manifest["digest"],
                                  "training_seed": config.seed, "initialization_seed": config.seed + 101,
                                  "optimization_seed": config.seed + 202,
                                  "data_order_seed_offset": 10_000, "dtype": "float32"}
        model = optimizer = scheduler = ema = payload = train_loss = validation_loss = None
        try:
            if family == "tokenizer":
                model = VQAutoencoder(run.config.tokenizer).to(run.device)
                perceptual = load_lpips(run, run.device)
                perceptual.requires_grad_(False).eval()
                train_data = ImageDataset(run.root, manifest["splits"]["train"], run.config.data.image_size)
                validation_data = ImageDataset(run.root, manifest["splits"]["validation"], run.config.data.image_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas,
                                             eps=config.epsilon, weight_decay=config.weight_decay)
            else:
                cache = load_token_cache(run, "train")
                validation_cache = load_token_cache(run, "validation")
                train_data, validation_data = TokenDataset(cache), TokenDataset(validation_cache)
                metadata.update({"codec_sha256": cache["codec_sha256"], "token_cache_identity": cache["identity"],
                                 "mean": cache["mean"], "std": cache["std"]})
                if family == "diffusion":
                    model = DiffusionUNet(run.config.diffusion, run.config.tokenizer.embedding_dim,
                                          len(run.config.data.classes), run.config.latent_size).to(run.device)
                    codec, _ = load_frozen_tokenizer(run, run.device)
                    schedule = DiffusionSchedule(run.config.diffusion).to(run.device)
                else:
                    model = ARTransformer(run.config.autoregressive, run.config.tokenizer.codebook_size,
                                          run.config.latent_size ** 2, len(run.config.data.classes)).to(run.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas,
                                              eps=config.epsilon, weight_decay=config.weight_decay)
            save_architecture(run, family, model)
            metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_multiplier(epoch, config))
            ema = ({name: value.detach().clone() for name, value in model.state_dict().items()}
                   if family == "diffusion" else None)
            start_epoch = 0
            global_step = 0
            best_validation = float("inf")
            total_seconds = 0.0
            seed_everything(config.seed + 202)  # optimization/dropout stream
            if state is not None:
                if state["metadata"].get("codec_sha256") != metadata.get("codec_sha256"):
                    raise RuntimeError("Resumed generator references a different tokenizer.")
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                if ema is not None:
                    ema = {name: value.to(run.device) for name, value in state["ema"].items()}
                start_epoch, global_step = state["epoch"], state["global_step"]
                best_validation = state["best_validation"]
                total_seconds = state["metadata"].get("training_seconds", 0.0)
                restore_rng(state["rng"])
            with TrainingTracker(run, family, purge_step=start_epoch + 1) as tracker:
                tracker.logger.info("Starting %s at completed epoch %d on %s", family, start_epoch, run.device)
                for epoch in tqdm(range(start_epoch, config.epochs), desc=f"Train {family}"):
                    # Reset private streams per epoch: interruption restarts exactly that epoch.
                    noise_train = torch.Generator().manual_seed(config.seed + 20_000 + epoch)
                    noise_validation = torch.Generator().manual_seed(run.config.diffusion.validation_seed)
                    if family == "tokenizer":
                        train_loss = validation_loss = partial(_loss_tokenizer, model, perceptual, run=run)
                    elif family == "diffusion":
                        train_loss = partial(_loss_diffusion, model, codec, schedule, cache, run=run, generator=noise_train)
                        validation_loss = partial(_loss_diffusion, model, codec, schedule, cache, run=run, generator=noise_validation)
                    else:
                        train_loss = validation_loss = partial(_loss_autoregressive, model, run=run)
                    synchronize(run.device)
                    started = time.perf_counter()
                    used_lr = optimizer.param_groups[0]["lr"]
                    training_metrics, steps = _run_epoch(
                        model, _loader(train_data, config, epoch, True), config, train_loss,
                        optimizer, ema, run.config.diffusion.ema_decay, f"{family} train {epoch + 1}",
                    )
                    global_step += steps
                    # Best diffusion checkpoint is selected using raw-model validation MSE.
                    validation_metrics, _ = _run_epoch(
                        model, _loader(validation_data, config, epoch, False), config, validation_loss,
                        None, None, 0.0, f"{family} validation {epoch + 1}",
                    )
                    synchronize(run.device)
                    epoch_seconds = time.perf_counter() - started
                    total_seconds += epoch_seconds
                    criterion = validation_metrics["reconstruction" if family == "tokenizer" else "loss"]
                    is_best = criterion < best_validation  # strict comparison keeps earliest ties
                    best_validation = min(best_validation, criterion)
                    scheduler.step()
                    metrics = {**{f"train/{key}": value for key, value in training_metrics.items()},
                               **{f"validation/{key}": value for key, value in validation_metrics.items()},
                               "learning_rate": used_lr, "epoch_seconds": epoch_seconds,
                               "training_seconds": total_seconds, "global_step": global_step,
                               "train_samples": len(train_data), "validation_samples": len(validation_data)}
                    metrics.update({f"memory/{key}": value for key, value in memory_snapshot(run.device).items()
                                    if isinstance(value, (int, float))})
                    metadata.update({"training_seconds": total_seconds, "next_data_epoch": epoch + 1,
                                     "validation_selection": "raw_model", "noise_epoch_seed": config.seed + 20_000 + epoch,
                                     "validation_noise_seed": run.config.diffusion.validation_seed})
                    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "ema": ema,
                               "epoch": epoch + 1, "global_step": global_step, "rng": capture_rng(run.device),
                               "best_validation": best_validation, "metadata": metadata}
                    save_epoch(run, family, payload, metrics, is_best=is_best, final=epoch + 1 == config.epochs)
                    tracker.log(epoch + 1, metrics)
            return {"family": family, "completed_epochs": config.epochs, "status": "complete",
                    "directory": str(run.model_dir(family)), "training_seconds": total_seconds}

        except BaseException as error:
            # Notebook tracebacks otherwise retain failed forward graphs and accelerator tensors.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            train_loss = validation_loss = payload = state = None
            model = optimizer = scheduler = codec = perceptual = ema = schedule = None
            gc.collect()
            # A damaged device must not replace the original training exception.
            with suppress(RuntimeError):
                if torch.device(run.device).type == "mps":
                    torch.mps.empty_cache()
                elif torch.device(run.device).type == "cuda":
                    torch.cuda.empty_cache()


def train_tokenizer(run: Any) -> dict[str, Any]:
    if dist.is_initialized():
        return _train_tokenizer_group(run)
    return _train(run, "tokenizer")


def _tokenizer_barrier() -> None:
    """Every UUID-isolated NCCL worker owns its local CUDA device zero."""
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[0])
    else:
        dist.barrier()


def _tokenizer_progress(rank: int, world_size: int, message: str) -> None:
    print(f"[tokenizer rank {rank}/{world_size}] {message}", flush=True)


def train_tokenizer_distributed(run: Any, rank: int, world_size: int,
                                init_method: str, backend: str = "nccl") -> dict[str, Any]:
    """Worker entry point; the launcher starts and supervises all rank processes.

    Production workers each receive one CUDA UUID before importing PyTorch.
    Gloo is supported solely for offline CPU correctness tests.
    """
    if dist.is_initialized():
        raise RuntimeError("The tokenizer worker must own a new process group.")
    if world_size != 2 or rank not in (0, 1):
        raise ValueError("This experiment uses exactly two tokenizer ranks.")
    device = torch.device(run.device)
    initialization: dict[str, Any] = {}
    if backend == "nccl":
        if device.type != "cuda" or torch.cuda.device_count() != 1:
            raise RuntimeError("Each NCCL tokenizer rank requires exactly one UUID-isolated CUDA device.")
        assigned = run.hardware["assignments"]["tokenizer"]
        if len(assigned) != world_size or os.environ.get("AFG_WORKER_UUIDS") != assigned[rank]:
            raise RuntimeError("Tokenizer rank order differs from the saved GPU UUID assignment.")
        torch.cuda.set_device(0)
        # Global rank one still owns local cuda:0 after UUID masking.
        initialization["device_id"] = torch.device("cuda", 0)
    elif backend != "gloo" or device.type != "cpu":
        raise ValueError("Use NCCL for CUDA or Gloo for synthetic CPU tests.")
    try:
        _tokenizer_progress(rank, world_size, f"Initializing {backend} process group on "
                            f"{initialization.get('device_id', device)}.")
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank,
                                world_size=world_size, timeout=timedelta(minutes=30), **initialization)
        _tokenizer_progress(rank, world_size, "Process group initialized; preparing the run.")
        return _train_tokenizer_group(run)
    finally:
        # No final barrier: on failure the launcher terminates a blocked peer.
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_tokenizer_group(run: Any) -> dict[str, Any]:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    topology = {"world_size": world_size, "backend": str(dist.get_backend()),
                "gpu_uuids": (getattr(run, "hardware", None) or {}).get("assignments", {}).get("tokenizer", [])}
    model = wrapped = optimizer = scheduler = perceptual = state = payload = train_loss = None
    with run.training_lock("tokenizer") if rank == 0 else nullcontext():
        try:
            run.verify()
            config = run.config.tokenizer.training
            # Rank zero reconciles interrupted exports/metrics before any peer reads.
            if rank == 0:
                state = load_training_state(run, "tokenizer")
            _tokenizer_progress(rank, world_size, "Waiting for checkpoint reconciliation barrier.")
            _tokenizer_barrier()
            _tokenizer_progress(rank, world_size, "Checkpoint reconciliation barrier complete.")
            if rank != 0:
                state = read_training_state(run, "tokenizer")
            if state is not None:
                restore_rank_state(state, topology, rank, lambda _: None)
                if state["epoch"] >= config.epochs:
                    return {"family": "tokenizer", "completed_epochs": state["epoch"], "status": "reused",
                            "directory": str(run.model_dir("tokenizer")), "world_size": world_size}
            _assert_mutable(run, "tokenizer")
            _tokenizer_progress(rank, world_size, "Loading prepared data manifest.")
            manifest = prepare_data(run)
            _tokenizer_progress(rank, world_size, "Data manifest ready; constructing tokenizer and LPIPS.")
            torch.set_num_threads(run.config.runtime.cpu_threads)
            seed_everything(config.seed + 101)
            model = VQAutoencoder(run.config.tokenizer).to(run.device)
            perceptual = load_lpips(run, run.device).requires_grad_(False).eval()
            train_data = ImageDataset(run.root, manifest["splits"]["train"], run.config.data.image_size)
            validation_data = ImageDataset(run.root, manifest["splits"]["validation"], run.config.data.image_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas,
                                         eps=config.epsilon, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_multiplier(epoch, config))
            start_epoch = global_step = 0
            best_validation, total_seconds = float("inf"), 0.0
            metadata = {"family": "tokenizer", "data_digest": manifest["digest"],
                        "training_seed": config.seed, "initialization_seed": config.seed + 101,
                        "optimization_seed_by_rank": [config.seed + 202 + index for index in range(world_size)],
                        "data_order_seed_offset": 10_000, "dtype": "float32",
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "execution_topology": topology}
            if state is not None:
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                start_epoch, global_step = state["epoch"], state["global_step"]
                best_validation = state["best_validation"]
                total_seconds = state["metadata"].get("training_seconds", 0.0)
            if rank == 0:
                save_architecture(run, "tokenizer", model)
            device_ids = [0] if torch.device(run.device).type == "cuda" else None
            # No running-statistic buffers exist in this GroupNorm architecture.
            # Disabling broadcasts permits unequal numbers of local microbatches.
            _tokenizer_progress(rank, world_size, "Models ready; initializing DistributedDataParallel.")
            wrapped = DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False)
            _tokenizer_progress(rank, world_size, "DistributedDataParallel ready.")
            seed_everything(config.seed + 202 + rank)
            if state is not None:
                restore_rank_state(state, topology, rank, restore_rng)
            tracker_context = TrainingTracker(run, "tokenizer", purge_step=start_epoch + 1) if rank == 0 else nullcontext()
            with tracker_context as tracker:
                if tracker is not None:
                    tracker.logger.info("Starting tokenizer at epoch %d using %s", start_epoch, topology)
                for epoch in tqdm(range(start_epoch, config.epochs), desc="Train tokenizer", disable=rank != 0):
                    train_loss = partial(_loss_tokenizer, wrapped, perceptual, run=run)
                    synchronize(run.device)
                    _tokenizer_barrier()
                    _tokenizer_progress(rank, world_size, f"Epoch {epoch + 1}/{config.epochs}: starting training.")
                    started = time.perf_counter()
                    used_lr = optimizer.param_groups[0]["lr"]
                    training_metrics, steps = run_distributed_epoch(
                        wrapped, train_data, config, epoch, train_loss, optimizer,
                        f"tokenizer train {epoch + 1}")
                    global_step += steps
                    validation_result: list[Any] = [None]
                    if rank == 0:
                        # Call the unwrapped module: validation has no rank peers.
                        validation_metrics, _ = _run_epoch(
                            model, _loader(validation_data, config, epoch, False), config,
                            partial(_loss_tokenizer, model, perceptual, run=run), None, None, 0.0,
                            f"tokenizer validation {epoch + 1}")
                        synchronize(run.device)
                        validation_result[0] = (validation_metrics, time.perf_counter() - started)
                    dist.broadcast_object_list(validation_result, src=0)
                    validation_metrics, epoch_seconds = validation_result[0]
                    total_seconds += epoch_seconds
                    criterion = validation_metrics["reconstruction"]
                    is_best = criterion < best_validation
                    best_validation = min(best_validation, criterion)
                    scheduler.step()
                    rng_by_rank = gather_rank_states(capture_rng(run.device))
                    if rank == 0:
                        metrics = {**{f"train/{key}": value for key, value in training_metrics.items()},
                                   **{f"validation/{key}": value for key, value in validation_metrics.items()},
                                   "learning_rate": used_lr, "epoch_seconds": epoch_seconds,
                                   "training_seconds": total_seconds, "global_step": global_step,
                                   "train_samples": len(train_data), "validation_samples": len(validation_data),
                                   "world_size": world_size, "global_effective_batch": config.effective_batch}
                        metrics.update({f"memory/{key}": value for key, value in memory_snapshot(run.device).items()
                                        if isinstance(value, (int, float))})
                        metadata.update({"training_seconds": total_seconds, "next_data_epoch": epoch + 1,
                                         "validation_selection": "raw_model"})
                        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scheduler": scheduler.state_dict(), "ema": None, "epoch": epoch + 1,
                                   "global_step": global_step, "rng": rng_by_rank[0],
                                   "distributed": {"topology": topology, "rng_by_rank": rng_by_rank},
                                   "best_validation": best_validation, "metadata": metadata}
                        save_epoch(run, "tokenizer", payload, metrics, is_best, epoch + 1 == config.epochs)
                        tracker.log(epoch + 1, metrics)
                    # Peers cannot start another epoch before its predecessor commits.
                    _tokenizer_barrier()
            return {"family": "tokenizer", "completed_epochs": config.epochs, "status": "complete",
                    "directory": str(run.model_dir("tokenizer")), "training_seconds": total_seconds,
                    "world_size": world_size}
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            train_loss = payload = state = model = wrapped = optimizer = scheduler = perceptual = None
            gc.collect()
            with suppress(RuntimeError):
                if torch.device(run.device).type == "cuda":
                    torch.cuda.empty_cache()


def train_diffusion(run: Any) -> dict[str, Any]:
    return _train(run, "diffusion")


def train_autoregressive(run: Any) -> dict[str, Any]:
    return _train(run, "autoregressive")
