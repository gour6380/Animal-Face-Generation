"""Frozen tokenizer selection, validation diagnostics and immutable token caches."""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import load_model_checkpoint
from .data import ImageDataset
from .data import load_prepared_data as prepare_data
from .io import atomic_json, digest_object, read_json, sha256_file
from .models.tokenizer import VQAutoencoder
from .tracking import TrainingTracker
from .weights import load_lpips


def _checkpoint_path(run: Any) -> Path:
    """Resolve the canonical selected codec without relying on current training state."""
    return run.model_dir("tokenizer") / "best_validation.pt"


def _load_selected(run: Any, device: torch.device | str) -> tuple[VQAutoencoder, str]:
    state = load_model_checkpoint(run, "tokenizer", kind="best_validation")
    model = VQAutoencoder(run.config.tokenizer).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval()
    return model, sha256_file(_checkpoint_path(run))


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round()
    image = array.to(torch.uint8).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def evaluate_reconstructions(run: Any) -> dict[str, Any]:
    """Evaluate only development-validation images with the selected tokenizer."""
    with run.lock():
        run.verify()
        manifest = prepare_data(run)
        output = run.path / "evaluation" / "reconstructions"
        checkpoint_hash = sha256_file(_checkpoint_path(run))
        identity = digest_object({"codec_sha256": checkpoint_hash, "data": manifest["digest"]})
        metric_path = output / "metrics.json"
        if metric_path.exists():
            previous = read_json(metric_path)
            if previous["identity"] != identity:
                raise RuntimeError("Reconstruction identity changed; create a fresh run.")
            for artifact, expected_hash in previous["artifacts"].items():
                if sha256_file(output / artifact) != expected_hash:
                    raise RuntimeError("Reconstruction artifact checksum mismatch.")
            return previous
        model, checkpoint_hash = _load_selected(run, run.device)
        perceptual = load_lpips(run, run.device)
        dataset = ImageDataset(run.root, manifest["splits"]["validation"], run.config.data.image_size)
        pin_memory = torch.device(run.device).type == "cuda"
        loader = DataLoader(dataset, batch_size=run.config.runtime.tokenizer_inference_batch, shuffle=False,
                            num_workers=0, pin_memory=pin_memory, generator=torch.Generator().manual_seed(3001))
        count = 0
        totals = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0}
        frequencies = torch.zeros(run.config.tokenizer.codebook_size, dtype=torch.int64)
        selected_per_class = {label: 0 for label in range(len(run.config.data.classes))}
        selected: list[dict[str, Any]] = []
        output.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Validation reconstruction"):
                original = batch["image"].to(run.device, non_blocking=pin_memory)
                result = model(original)
                reconstructed = result["reconstruction"]
                l1 = (reconstructed - original).abs().flatten(1).mean(1)
                lpips = perceptual(reconstructed, original).flatten(1).mean(1)
                mse = ((reconstructed - original) * 0.5).square().flatten(1).mean(1)
                psnr = -10 * torch.log10(mse.clamp_min(1e-12))
                for key, value in (("l1", l1), ("lpips", lpips), ("psnr", psnr)):
                    totals[key] += value.sum().item()
                frequencies += torch.bincount(result["tokens"].flatten().cpu(), minlength=len(frequencies))
                count += len(original)
                for index, raw_label in enumerate(batch["label"]):
                    label = int(raw_label)
                    if selected_per_class[label] >= 4:
                        continue
                    item_id = str(batch["id"][index])
                    stem = f"{label}_{selected_per_class[label]:02d}"
                    _save_image(original[index], output / f"{stem}_original.png")
                    _save_image(reconstructed[index], output / f"{stem}_reconstructed.png")
                    selected.append({"id": item_id, "label": label, "stem": stem})
                    selected_per_class[label] += 1
        if not count:
            raise ValueError("The validation split is empty.")
        probabilities = frequencies.double() / frequencies.sum()
        active = probabilities > 0
        perplexity = torch.exp(-(probabilities[active] * probabilities[active].log()).sum()).item()
        ordered = sorted(selected, key=lambda item: (item["label"], item["stem"]))
        side = run.config.data.image_size
        grid = Image.new("RGB", (side * 8, side * len(run.config.data.classes)))
        for item in ordered:
            row = item["label"]
            col = int(item["stem"].split("_")[1]) * 2
            for offset, suffix in enumerate(("original", "reconstructed")):
                with Image.open(output / f"{item['stem']}_{suffix}.png") as picture:
                    grid.paste(picture, ((col + offset) * side, row * side))
        grid.save(output / "grid.png")
        metrics = {key: value / count for key, value in totals.items()}
        metrics.update({"objective": metrics["l1"] + run.config.tokenizer.perceptual_weight * metrics["lpips"],
                        "count": count, "codebook_usage": active.sum().item() / len(frequencies),
                        "used_codes": int(active.sum()), "perplexity": perplexity,
                        "codec_sha256": checkpoint_hash, "identity": identity, "split": "validation",
                        "selected": ordered, "psnr_epsilon": 1e-12,
                        "artifacts": {path.name: sha256_file(path) for path in sorted(output.glob("*.png"))}})
        atomic_json(metric_path, metrics)
        with TrainingTracker(run, "tokenizer") as tracker:
            tracker.writer.add_image("validation/selected_reconstruction_grid", np.asarray(grid),
                                     global_step=0, dataformats="HWC")
        del model, perceptual
        gc.collect()
        return metrics


def freeze_tokenizer(run: Any) -> dict[str, Any]:
    """Freeze the earliest minimum-validation-reconstruction codec after diagnostics."""
    with run.lock():
        run.verify()
        data = prepare_data(run)
        final_state = load_model_checkpoint(run, "tokenizer", kind="final")
        if final_state["epoch"] != run.config.tokenizer.training.epochs:
            raise RuntimeError("Finish every configured tokenizer epoch before freezing the shared representation.")
        del final_state
        metrics_path = run.path / "evaluation" / "reconstructions" / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError("Evaluate validation reconstructions before freezing the tokenizer.")
        checkpoint_hash = sha256_file(_checkpoint_path(run))
        metrics = read_json(metrics_path)
        if metrics["codec_sha256"] != checkpoint_hash:
            raise RuntimeError("Reconstruction diagnostics do not match the selected tokenizer.")
        # Verify the checkpoint against its committed identity before registering it.
        state = load_model_checkpoint(run, "tokenizer", kind="best_validation")
        manifest_path = run.path / "data" / "tokenizer.json"
        result = {"codec_sha256": checkpoint_hash, "checkpoint": str(_checkpoint_path(run).relative_to(run.path)),
                  "data_digest": data["digest"], "selected_epoch": state["epoch"],
                  "selection": "minimum validation L1 + perceptual_weight * LPIPS; earliest tie",
                  "reconstruction_metrics_sha256": sha256_file(metrics_path),
                  "codebook_size": run.config.tokenizer.codebook_size,
                  "embedding_dim": run.config.tokenizer.embedding_dim,
                  "latent_size": run.config.latent_size}
        if manifest_path.exists():
            existing = read_json(manifest_path)
            if any(existing.get(key) != value for key, value in result.items()):
                raise RuntimeError("Frozen tokenizer identity changed; create a fresh run.")
            return existing
        atomic_json(manifest_path, result)
        return result


def load_frozen_tokenizer(run: Any, device: torch.device | str = "cpu") -> tuple[VQAutoencoder, dict[str, Any]]:
    manifest = read_json(run.path / "data" / "tokenizer.json")
    path = run.path / manifest["checkpoint"]
    if sha256_file(path) != manifest["codec_sha256"]:
        raise RuntimeError("Frozen tokenizer checkpoint checksum mismatch.")
    # Loading from state_dict is explicit and does not import or pickle a model object.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = VQAutoencoder(run.config.tokenizer).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval()
    return model, manifest


def _write_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def latent_moments(vectors: torch.Tensor, frequencies: torch.Tensor) -> tuple[float, float]:
    """Population moments weighted by observed train tokens, including all channels."""
    vectors = vectors.detach().cpu().double()
    frequencies = frequencies.detach().cpu().double()
    if vectors.ndim != 2 or frequencies.shape != (vectors.shape[0],):
        raise ValueError("Each code vector must have one observed-token frequency.")
    if (frequencies < 0).any() or frequencies.sum() <= 0:
        raise ValueError("Code frequencies must be nonnegative with positive total count.")
    weights = frequencies.unsqueeze(1)
    denominator = frequencies.sum().item() * vectors.shape[1]
    mean = (vectors * weights).sum().item() / denominator
    variance = ((vectors - mean).square() * weights).sum().item() / denominator
    std = math.sqrt(variance)
    if not math.isfinite(std) or std <= 1e-8:
        raise RuntimeError("Training latent variance is degenerate; inspect tokenizer diagnostics.")
    return mean, std


def cache_tokens(run: Any) -> dict[str, Any]:
    """Materialize only train/validation IDs; fit one scalar normalization on train."""
    with run.lock():
        run.verify()
        data = prepare_data(run)
        codec = read_json(run.path / "data" / "tokenizer.json")
        if codec["data_digest"] != data["digest"]:
            raise RuntimeError("Frozen tokenizer was prepared with different data.")
        directory = run.path / "data" / "tokens"
        directory.mkdir(parents=True, exist_ok=True)
        identity = digest_object({"codec_sha256": codec["codec_sha256"], "data_digest": data["digest"]})
        destination = directory / "manifest.json"
        if destination.exists():
            previous = read_json(destination)
            if previous["identity"] != identity:
                raise RuntimeError("Token cache identity changed; create a fresh run.")
            for split in ("train", "validation"):
                load_token_cache(run, split)
            atomic_json(run.path / "data" / "tokenizer.json", {**codec, "mean": previous["mean"],
                        "std": previous["std"], "token_cache_identity": identity})
            return previous
        model, codec = load_frozen_tokenizer(run, run.device)
        records: dict[str, Any] = {}
        frequencies = torch.zeros(run.config.tokenizer.codebook_size, dtype=torch.int64)
        inference_batch = run.config.runtime.tokenizer_inference_batch
        pin_memory = torch.device(run.device).type == "cuda"
        with torch.inference_mode():
            for split in ("train", "validation"):
                dataset = ImageDataset(run.root, data["splits"][split], run.config.data.image_size)
                loader = DataLoader(dataset, batch_size=inference_batch,
                                    shuffle=False, num_workers=0, pin_memory=pin_memory,
                                    generator=torch.Generator().manual_seed(3002))
                tokens = np.empty((len(dataset), run.config.latent_size, run.config.latent_size), dtype=np.int64)
                labels = np.empty(len(dataset), dtype=np.int64)
                ids: list[str] = []
                offset = 0
                for batch in tqdm(loader, desc=f"Tokenize {split}"):
                    encoded = model.encode(batch["image"].to(run.device, non_blocking=pin_memory)).cpu()
                    end = offset + len(encoded)
                    tokens[offset:end] = encoded.numpy()
                    labels[offset:end] = batch["label"].numpy()
                    ids.extend(str(item) for item in batch["id"])
                    if split == "train":
                        frequencies += torch.bincount(encoded.flatten(), minlength=len(frequencies))
                    offset = end
                if not offset:
                    raise ValueError(f"The {split} split is empty.")
                _write_numpy(directory / f"{split}_tokens.npy", tokens)
                _write_numpy(directory / f"{split}_labels.npy", labels)
                atomic_json(directory / f"{split}_ids.json", ids)
                files = [f"{split}_tokens.npy", f"{split}_labels.npy", f"{split}_ids.json"]
                records[split] = {"count": offset, "files": {name: sha256_file(directory / name) for name in files}}
            # Count-weighted moments over every observed training embedding component.
            code_ids = torch.arange(run.config.tokenizer.codebook_size, device=run.device).view(-1, 1, 1)
            vectors = model.lookup(code_ids).squeeze(-1).squeeze(-1).cpu().double()
            mean, std = latent_moments(vectors, frequencies)
        result = {"identity": identity, "codec_sha256": codec["codec_sha256"],
                  "data_digest": data["digest"], "mean": mean, "std": std,
                  "inference_batch": inference_batch,
                  "normalization": "one scalar population mean/std over observed training embeddings",
                  "splits": records}
        atomic_json(destination, result)
        atomic_json(run.path / "data" / "tokenizer.json", {**codec, "mean": mean, "std": std,
                    "token_cache_identity": identity})
        del model
        gc.collect()
        return result


def load_token_cache(run: Any, split: str) -> dict[str, Any]:
    if split not in ("train", "validation"):
        raise ValueError("Training caches are restricted to train/validation.")
    directory = run.path / "data" / "tokens"
    manifest = read_json(directory / "manifest.json")
    codec = read_json(run.path / "data" / "tokenizer.json")
    data = read_json(run.path / "data" / "manifest.json")
    if (manifest["codec_sha256"] != codec["codec_sha256"] or manifest["data_digest"] != data["digest"]
            or sha256_file(run.path / codec["checkpoint"]) != codec["codec_sha256"]):
        raise RuntimeError("Token cache is stale or references a modified checkpoint.")
    metadata = manifest["splits"][split]
    for name, expected in metadata["files"].items():
        if sha256_file(directory / name) != expected:
            raise RuntimeError(f"Token cache checksum mismatch: {name}")
    tokens = torch.from_numpy(np.load(directory / f"{split}_tokens.npy", allow_pickle=False))
    labels = torch.from_numpy(np.load(directory / f"{split}_labels.npy", allow_pickle=False))
    ids = read_json(directory / f"{split}_ids.json")
    if len(tokens) != metadata["count"] or len(labels) != len(tokens) or len(ids) != len(tokens):
        raise RuntimeError("Token cache lengths disagree.")
    expected_records = data["splits"][split]
    expected_ids = [str(record["id"]) for record in expected_records]
    expected_labels = torch.tensor([record["label"] for record in expected_records], dtype=torch.int64)
    if ids != expected_ids or labels.dtype != torch.int64 or labels.shape != expected_labels.shape:
        raise RuntimeError("Token cache labels/record IDs do not match the registered split ordering.")
    if not torch.equal(labels, expected_labels):
        raise RuntimeError("Token cache labels/record IDs do not match the registered split ordering.")
    if tokens.dtype != torch.int64 or tokens.shape[1:] != (run.config.latent_size, run.config.latent_size):
        raise RuntimeError("Token cache shape/dtype mismatch.")
    if tokens.min() < 0 or tokens.max() >= run.config.tokenizer.codebook_size:
        raise RuntimeError("Token cache contains out-of-codebook values.")
    return {"tokens": tokens, "labels": labels, "ids": ids, "mean": manifest["mean"],
            "std": manifest["std"], "codec_sha256": manifest["codec_sha256"],
            "identity": manifest["identity"]}
