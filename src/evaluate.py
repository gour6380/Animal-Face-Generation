"""Frozen test protocol, feature caches and finite-sample generation diagnostics."""
from __future__ import annotations

import hashlib
import io
import os
import random
import re
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import EvaluationConfig
from .io import atomic_json, digest_object, read_json, sha256_file


def _locked(function: Any) -> Any:
    @wraps(function)
    def wrapped(run: Any, *args: Any, **kwargs: Any) -> Any:
        with run.lock():
            return function(run, *args, **kwargs)
    return wrapped


def _analysis_path(run: Any, analysis: str) -> Path:
    if not re.fullmatch(r"analysis_[0-9]{3,}", analysis):
        raise ValueError("Expected an analysis_XXX revision name.")
    return run.path / "evaluation" / analysis


def _source_hashes(run: Any) -> dict[str, str]:
    return {str(path.relative_to(run.root)): sha256_file(path)
            for path in sorted((run.root / "src").rglob("*.py"))}


def _weight_identity(run: Any) -> dict[str, Any]:
    path = run.path / "data" / "weights.json"
    manifest = read_json(path)
    for name, record in manifest["weights"].items():
        if sha256_file(run.root / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Frozen feature-network weights changed: {name}")
    return {"manifest_sha256": sha256_file(path),
            "weights": {name: record["sha256"] for name, record in manifest["weights"].items()}}


@_locked
def freeze_evaluation(run: Any, evaluation_config: EvaluationConfig | None = None) -> str:
    """Freeze final checkpoints before test access; settings changes create new revisions."""
    from .checkpoints import load_model_checkpoint
    from .data import load_prepared_data
    from .models.diffusion import ddim_timesteps
    from .sample import setting_spec, verify_codec_metadata

    run.verify()
    config = evaluation_config or run.config.evaluation
    if min(config.max_per_class, config.generation_batch, config.feature_batch,
           config.kid_subsets, config.kid_class_subset, config.kid_overall_subset) < 1:
        raise ValueError("Evaluation sizes must be positive.")
    if config.kid_class_subset < 2 or config.kid_overall_subset < 2:
        raise ValueError("Unbiased KID requires at least two samples per subset.")
    if 50 not in config.ddim_steps or 1.0 not in config.temperatures:
        raise ValueError("Primary settings DDIM-50 and temperature 1.0 are mandatory.")
    if any(not 2 <= step <= run.config.diffusion.timesteps for step in config.ddim_steps):
        raise ValueError("Invalid DDIM step count.")
    if config.precision_recall_k < 1 or config.benchmark_repetitions < 2 or config.benchmark_warmups < 0:
        raise ValueError("Invalid metric or benchmark protocol.")
    if any(batch < 1 for batch in config.benchmark_batches):
        raise ValueError("Benchmark batches must be positive.")
    manifest = load_prepared_data(run)
    token_manifest_path = run.path / "data" / "tokenizer.json"
    tokenizer = read_json(token_manifest_path)
    if "mean" not in tokenizer or "std" not in tokenizer:
        raise RuntimeError("Token caching and train-only latent statistics must be complete.")
    checkpoints = {}
    for family in ("tokenizer", "diffusion", "autoregressive"):
        checkpoint = load_model_checkpoint(run, family, kind="final")
        if checkpoint["epoch"] != getattr(run.config, family).training.epochs:
            raise RuntimeError(f"{family} has not completed the fixed training budget.")
        if family != "tokenizer":
            verify_codec_metadata(checkpoint["metadata"], tokenizer)
        path = run.model_dir(family) / "final.pt"
        checkpoints[family] = {"path": str(path.relative_to(run.path)), "sha256": sha256_file(path),
                               "epoch": checkpoint["epoch"], "parameters": checkpoint["metadata"].get("parameters"),
                               "training_seconds": checkpoint["metadata"].get("training_seconds")}
    selected_codec_path = run.path / tokenizer["checkpoint"]
    if sha256_file(selected_codec_path) != tokenizer["codec_sha256"]:
        raise RuntimeError("Selected codec checkpoint changed.")
    settings = {f"ddim_{step}": setting_spec("diffusion", f"ddim_{step}") for step in config.ddim_steps}
    settings.update({f"temperature_{float(temp)}": setting_spec("autoregressive", f"temperature_{float(temp)}")
                     for temp in config.temperatures})
    fixed = {"checkpoints": checkpoints, "sources": _source_hashes(run),
             "model_config_sha256": digest_object({k: v for k, v in run.config.to_dict().items()
                                                  if k != "evaluation"}),
             "data_manifest_sha256": digest_object(manifest),
             "feature_weights": _weight_identity(run),
             "tokenizer_manifest_sha256": sha256_file(token_manifest_path),
             "selected_codec": {"path": tokenizer["checkpoint"], "sha256": tokenizer["codec_sha256"]}}
    candidate = {"protocol_version": 1, "fixed": fixed, "evaluation_config": asdict(config),
                 "ddim_timestep_schedules": {f"ddim_{step}": ddim_timesteps(run.config.diffusion.timesteps, step)
                                             for step in config.ddim_steps},
                 "settings": settings, "primary_settings": ["ddim_50", "temperature_1.0"],
                 "checkpoint_selection": "codec best development-validation after its full budget; priors final epoch; diffusion final EMA",
                 "limitations": ["one training seed", "shared learned codec", "small held-out reference",
                                  "KID subset standard deviation is not training-seed uncertainty"]}
    with run.lock():
        directory = run.path / "evaluation"
        revisions = sorted(path for path in directory.glob("analysis_*") if path.is_dir())
        for revision in revisions:
            path = revision / "freeze.json"
            if not path.exists():
                raise RuntimeError(f"Incomplete analysis directory: {revision}")
            existing = read_json(path)
            if existing["fixed"] != fixed:
                raise RuntimeError("Models, data or code changed after the first test freeze. Create a new run.")
            if digest_object(existing) == digest_object(candidate):
                return revision.name
        next_number = max((int(path.name.split("_")[-1]) for path in revisions), default=0) + 1
        analysis = f"analysis_{next_number:03d}"
        target = _analysis_path(run, analysis)
        target.mkdir(parents=True, exist_ok=False)
        atomic_json(target / "freeze.json", candidate)
        return analysis


def load_freeze(run: Any, analysis: str) -> dict[str, Any]:
    """Validate all frozen identities before any held-out or generation operation."""
    run.verify()
    freeze = read_json(_analysis_path(run, analysis) / "freeze.json")
    fixed = freeze["fixed"]
    if fixed["sources"] != _source_hashes(run):
        raise RuntimeError("Source changed after evaluation freeze.")
    if fixed["feature_weights"] != _weight_identity(run):
        raise RuntimeError("Frozen evaluator weight identity changed.")
    config_hash = digest_object({k: v for k, v in run.config.to_dict().items() if k != "evaluation"})
    if fixed["model_config_sha256"] != config_hash:
        raise RuntimeError("Model/training configuration changed after evaluation freeze.")
    for family, checkpoint in fixed["checkpoints"].items():
        if sha256_file(run.path / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"Frozen {family} checkpoint changed.")
    if sha256_file(run.path / "data" / "tokenizer.json") != fixed["tokenizer_manifest_sha256"]:
        raise RuntimeError("Frozen tokenizer identity changed.")
    if sha256_file(run.path / fixed["selected_codec"]["path"]) != fixed["selected_codec"]["sha256"]:
        raise RuntimeError("Selected codec checkpoint changed after freeze.")
    return freeze


def balanced_records(records: list[dict[str, Any]], n_per_class: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    for label in range(3):
        group = sorted((row for row in records if row["label"] == label), key=lambda row: row["id"])
        if len(group) < n_per_class:
            raise ValueError("Requested balanced reference exceeds an available class.")
        selected.extend(random.Random(seed + label).sample(group, n_per_class))
    return sorted(selected, key=lambda row: (row["label"], row["id"]))


def prepare_reference(run: Any, analysis: str) -> dict[str, Any]:
    freeze = load_freeze(run, analysis)
    return _prepare_reference(run, analysis, freeze)


def _prepare_reference(run: Any, analysis: str, freeze: dict[str, Any]) -> dict[str, Any]:
    """Reuse the caller's verified freeze within one locked operation."""
    from .data import load_prepared_data
    data = load_prepared_data(run)
    if digest_object(data) != freeze["fixed"]["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed after freeze.")
    cfg = freeze["evaluation_config"]
    test = data["splits"]["test"]
    # n_per_class is inventoried before any training and may only be reduced explicitly.
    n = min(int(data["n_per_class"]), int(cfg["max_per_class"]))
    if n < 2:
        raise ValueError("Per-class distribution metrics require at least two held-out examples.")
    result = {"freeze_sha256": digest_object(freeze), "n_per_class": n,
              "records": balanced_records(test, n, cfg["reference_seed"]),
              "effective_metric_sizes": {"kid_class_subset": min(cfg["kid_class_subset"], n),
                                          "kid_overall_subset": min(cfg["kid_overall_subset"], 3 * n),
                                          "precision_recall_class_k": min(cfg["precision_recall_k"], n - 1),
                                          "precision_recall_overall_k": min(cfg["precision_recall_k"], 3 * n - 1)},
              "population": "equal cat/dog/wildlife mixture from the upstream held-out partition"}
    path = _analysis_path(run, analysis) / "reference.json"
    with run.lock():
        if path.exists():
            existing = read_json(path)
            if existing != result:
                raise RuntimeError("Reference selection changed.")
            return existing
        atomic_json(path, result)
    return result


def kernel_inception_distance(real: np.ndarray, generated: np.ndarray, *, subset_size: int,
                              subsets: int, seed: int) -> dict[str, Any]:
    """Unbiased polynomial MMD in Inception features; negative estimates are valid."""
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("Expected feature matrices of the same width.")
    if subset_size < 2 or subset_size > min(len(real), len(generated)) or subsets < 1:
        raise ValueError("Invalid KID subset sizes.")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("Nonfinite evaluator features.")
    rng, estimates = np.random.default_rng(seed), []
    # The frozen protocol repeatedly samples the same modest populations. Compute
    # their kernels once instead of repeating 2048-dimensional products 100 times.
    # Bound memory for callers evaluating substantially larger populations.
    entries = len(real) ** 2 + len(generated) ** 2 + len(real) * len(generated)
    reuse = entries * np.dtype(np.float64).itemsize <= 128 * 1024**2
    reuse = reuse and entries <= 3 * subsets * subset_size**2
    kernels = None
    if reuse:
        width = real.shape[1]
        kernels = ((real @ real.T / width + 1) ** 3,
                   (generated @ generated.T / width + 1) ** 3,
                   (real @ generated.T / width + 1) ** 3)
    for _ in range(subsets):
        ix = rng.choice(len(real), subset_size, replace=False)
        iy = rng.choice(len(generated), subset_size, replace=False)
        if kernels is None:
            x, y = real[ix], generated[iy]
            xx, yy, xy = ((x @ x.T / x.shape[1] + 1) ** 3,
                          (y @ y.T / y.shape[1] + 1) ** 3,
                          (x @ y.T / x.shape[1] + 1) ** 3)
        else:
            xx, yy, xy = (kernels[0][np.ix_(ix, ix)], kernels[1][np.ix_(iy, iy)],
                          kernels[2][np.ix_(ix, iy)])
        within = ((xx.sum() - np.trace(xx)) + (yy.sum() - np.trace(yy))) / (subset_size * (subset_size - 1))
        estimates.append(float(within - 2 * xy.mean()))
    return {"mean": float(np.mean(estimates)), "subset_std": float(np.std(estimates)),
            "subsets": subsets, "subset_size": subset_size, "seed": seed,
            "uncertainty_meaning": "variation over finite-sample subsets; not a confidence interval or training-seed variability"}


def _radii(features: torch.Tensor, k: int, chunk: int = 128) -> torch.Tensor:
    values = []
    for start in range(0, len(features), chunk):
        end = min(start + chunk, len(features))
        distance = torch.cdist(features[start:end], features)
        distance[torch.arange(end - start), torch.arange(start, end)] = float("inf")
        values.append(distance.kthvalue(k, dim=1).values)
    return torch.cat(values)


def _inside(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor,
            chunk: int = 128) -> float:
    covered = 0
    for start in range(0, len(points), chunk):
        covered += int((torch.cdist(points[start:start + chunk], centers) <= radii[None, :]).any(dim=1).sum())
    return covered / len(points)


def feature_precision_recall(real: torch.Tensor, generated: torch.Tensor, k: int = 3) -> dict[str, Any]:
    """k-NN manifold diagnostic in the same 2048-dimensional Inception features."""
    real, generated = real.detach().float().cpu(), generated.detach().float().cpu()
    if min(len(real), len(generated)) <= k or k < 1:
        raise ValueError("Neighborhood k must be smaller than both sample counts.")
    return {"precision": _inside(generated, real, _radii(real, k)),
            "recall": _inside(real, generated, _radii(generated, k)), "k": k,
            "interpretation": "finite-sample Inception-space diagnostic; not semantic correctness"}


class _ReferenceImages(Dataset):
    def __init__(self, run: Any, records: list[dict[str, Any]]):
        from .data import ImageDataset
        self.dataset = ImageDataset(run.root, records, image_size=run.config.data.image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        from .sample import image_to_uint8
        return image_to_uint8(self.dataset[index]["image"])


class _GeneratedImages(Dataset):
    def __init__(self, directory: Path, records: list[dict[str, Any]]):
        self.directory, self.records = directory, records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.records[index]
        path = self.directory / row["file"]
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Generated image changed: {path}")
        with Image.open(io.BytesIO(encoded)) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)


@torch.inference_mode()
def _cached_features(run: Any, dataset: Dataset, identity: dict[str, Any], batch_size: int) -> torch.Tensor:
    from .weights import load_inception
    identity = {"input": identity, "extractor": "torch-fidelity/inception-v3-compat/2048",
                "weights": _weight_identity(run),
                "preprocessing": "uint8 RGB128; official extractor internal resize; no external normalization",
                "device": str(run.device), "dtype": "float32"}
    key = digest_object(identity)
    directory = run.path / "evaluation" / "feature_cache"
    path, metadata_path = directory / f"{key}.pt", directory / f"{key}.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata["identity"] != identity or sha256_file(path) != metadata["sha256"]:
            raise RuntimeError("Feature cache was modified.")
        return torch.load(path, map_location="cpu", weights_only=True)
    extractor = load_inception(run, str(run.device))
    chunks = []
    for images in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                       desc="Inception features", unit="batch"):
        output = extractor(images.to(run.device))
        features = output[0] if isinstance(output, (tuple, list)) else output
        if features.ndim != 2 or features.shape[1] != 2048 or not torch.isfinite(features).all():
            raise RuntimeError("Inception extractor returned invalid features.")
        chunks.append(features.detach().float().cpu())
    features = torch.cat(chunks)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(features, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(metadata_path, {"identity": identity, "sha256": sha256_file(path), "count": len(features)})
    return features


def nearest_training_examples(generated: torch.Tensor, train: torch.Tensor, chunk: int = 64) -> tuple[list[int], list[float]]:
    indexes, distances = [], []
    for start in range(0, len(generated), chunk):
        values, indices = torch.cdist(generated[start:start + chunk].float().cpu(), train.float().cpu()).min(dim=1)
        indexes.extend(indices.tolist())
        distances.extend(values.tolist())
    return indexes, distances


def evaluate_samples(run: Any, analysis: str, setting: str) -> dict[str, Any]:
    """Evaluate existing sample files; this never silently generates or selects checkpoints."""
    from .data import load_prepared_data
    freeze = load_freeze(run, analysis)
    reference = _prepare_reference(run, analysis, freeze)
    if setting not in freeze["settings"]:
        raise ValueError("Setting is not frozen.")
    directory = _analysis_path(run, analysis)
    samples_dir = directory / "samples" / setting
    sample_manifest = read_json(samples_dir / "manifest.json")
    if not sample_manifest["complete"]:
        raise RuntimeError("Sample generation is incomplete.")
    if sample_manifest["identity"]["freeze_sha256"] != digest_object(freeze):
        raise RuntimeError("Samples belong to another evaluation freeze.")
    cfg = freeze["evaluation_config"]
    records = sample_manifest["records"]
    if any(sum(row["label"] == label for row in records) != reference["n_per_class"] for label in range(3)):
        raise RuntimeError("Generated sample class counts violate the frozen protocol.")
    for row in records:
        if sha256_file(samples_dir / row["file"]) != row["sha256"]:
            raise RuntimeError("Generated sample changed.")
    path = directory / "metrics" / f"{setting}.json"
    identity = {"freeze_sha256": digest_object(freeze), "sample_manifest_sha256": digest_object(sample_manifest),
                "reference_sha256": digest_object(reference), "setting": setting}
    with run.lock():
        if path.exists():
            existing = read_json(path)
            if existing["identity"] != identity:
                raise RuntimeError("Metrics identity changed.")
            return existing
        real = _cached_features(run, _ReferenceImages(run, reference["records"]),
                                {"records": reference["records"]}, cfg["feature_batch"])
        fake = _cached_features(run, _GeneratedImages(samples_dir, records),
                                {"samples": sample_manifest}, cfg["feature_batch"])
        def metrics(a: torch.Tensor, b: torch.Tensor, size: int, seed: int, k: int) -> dict[str, Any]:
            return {"kid": kernel_inception_distance(a.numpy(), b.numpy(), subset_size=size,
                                                      subsets=cfg["kid_subsets"], seed=seed),
                    "precision_recall": feature_precision_recall(a, b, k)}
        sizes = reference["effective_metric_sizes"]
        result = {"identity": identity, "overall": metrics(real, fake, sizes["kid_overall_subset"], cfg["metric_seed"],
                                                            sizes["precision_recall_overall_k"]),
                  "per_class": {}, "n_per_class": reference["n_per_class"],
                  "class_adherence": "Qualitative review of fixed random grids; no classifier accuracy is claimed."}
        real_labels = torch.tensor([row["label"] for row in reference["records"]])
        fake_labels = torch.tensor([row["label"] for row in records])
        for label, name in enumerate(run.config.data.classes):
            result["per_class"][name] = metrics(real[real_labels == label], fake[fake_labels == label],
                                                sizes["kid_class_subset"], cfg["metric_seed"] + label + 1,
                                                sizes["precision_recall_class_k"])
        means = [entry["kid"]["mean"] for entry in result["per_class"].values()]
        result["macro_class_kid"] = {"mean": float(np.mean(means)), "between_class_std": float(np.std(means)),
                                     "meaning": "mean of class KIDs; between-class spread is not uncertainty"}
        data = load_prepared_data(run)
        train_records = data["splits"]["train"]
        train = _cached_features(run, _ReferenceImages(run, train_records),
                                 {"records": train_records}, cfg["feature_batch"])
        indexes, distances = nearest_training_examples(fake, train)
        nearest = [{"generated_file": row["file"], "training_id": train_records[index]["id"],
                    "training_relative_path": train_records[index]["relative_path"],
                    "inception_distance": distance}
                   for row, index, distance in zip(records, indexes, distances)]
        nearest_path = directory / "nearest_training" / f"{setting}.json"
        atomic_json(nearest_path, {"identity": identity, "records": nearest,
                                  "meaning": "Nearest-neighbor inspection; closeness is not proof of copying."})
        hashes = [row["sha256"] for row in records]
        result["exact_generated_duplicates"] = len(hashes) - len(set(hashes))
        result["nearest_training"] = {"path": str(nearest_path.relative_to(run.path)),
                                      "sha256": sha256_file(nearest_path),
                                      "minimum_distance": min(distances),
                                      "median_distance": float(np.median(distances))}
        atomic_json(path, result)
        return result
