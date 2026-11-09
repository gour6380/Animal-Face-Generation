"""Notebook figures and reports derived strictly from existing, verified artifacts."""
from __future__ import annotations

import json
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .evaluate import _analysis_path, balanced_records, load_freeze
from .io import atomic_bytes, atomic_json, digest_object, read_json, sha256_file


def _locked(function: Any) -> Any:
    """Notebook artifact operations share the run's reentrant single-writer lock."""
    @wraps(function)
    def wrapped(run: Any, *args: Any, **kwargs: Any) -> Any:
        with run.lock():
            run.verify()
            return function(run, *args, **kwargs)
    return wrapped


def _save_figure(figure: Any, path: Path) -> Any:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    atomic_bytes(path, buffer.getvalue())
    plt.close(figure)
    return figure


def _log_figure(run: Any, family: str, tag: str, figure: Any, step: int = 0) -> None:
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(run.path / "tensorboard" / family)) as writer:
        writer.add_figure(tag, figure, global_step=step, close=False)


@_locked
def show_training_samples(run: Any) -> Any:
    """Inspect twelve real training examples; never touch the held-out test images."""
    from .data import ImageDataset
    from .data import load_prepared_data as prepare_data
    manifest = prepare_data(run)
    selected = balanced_records(manifest["splits"]["train"], 4, run.config.data.split_seed + 10)
    dataset = ImageDataset(run.root, selected, image_size=run.config.data.image_size)
    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for index, axis in enumerate(axes.flat):
        item = dataset[index]
        axis.imshow(((item["image"].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
        axis.set_title(f"{run.config.data.display_classes[item['label']]} · {selected[index]['id']}", fontsize=7)
        axis.axis("off")
    figure.suptitle("Real training images — fixed selection")
    figure.tight_layout()
    return _save_figure(figure, run.path / "data" / "training_samples.png")


@_locked
def plot_training_history(run: Any, family: str) -> Any:
    if family not in ("tokenizer", "diffusion", "autoregressive"):
        raise ValueError("Unknown model family.")
    path = run.model_dir(family) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No completed training epochs: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Training history is empty.")
    # Training commits may wrap the payload as metrics; accept both documented history layouts.
    metrics = [row.get("metrics", row) for row in rows]
    epochs = [row.get("epoch", values.get("epoch", index + 1))
              for index, (row, values) in enumerate(zip(rows, metrics))]
    keys = [key for key in metrics[0] if key.startswith(("train/", "validation/"))
            and key not in ("train/seconds", "validation/seconds")]
    figure, axis = plt.subplots(figsize=(9, 4))
    for key in keys:
        if all(isinstance(item.get(key), (int, float)) for item in metrics):
            axis.plot(epochs, [item[key] for item in metrics], label=key)
    axis.set(xlabel="Completed epoch", ylabel="Within-model objective / diagnostic",
             title=f"{family}: training and development validation")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, run.model_dir(family) / "history.png")


@_locked
def show_reconstructions(run: Any) -> Any:
    directory = run.path / "evaluation" / "reconstructions"
    metrics = read_json(directory / "metrics.json")
    path = directory / "grid.png"
    if metrics.get("artifacts", {}).get("grid.png") != sha256_file(path):
        raise RuntimeError("Reconstruction grid changed.")
    figure, axis = plt.subplots(figsize=(13, 5))
    with Image.open(path) as image:
        axis.imshow(np.array(image))
    axis.axis("off")
    axis.set_title("Development validation — original / reconstruction pairs; selected frozen codec")
    figure.tight_layout()
    _log_figure(run, "tokenizer", "validation/reconstructions", figure)
    plt.close(figure)
    return figure


def _sample_records(run: Any, analysis: str, setting: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    freeze = load_freeze(run, analysis)
    if setting not in freeze["settings"]:
        raise ValueError("Setting was not frozen.")
    directory = _analysis_path(run, analysis) / "samples" / setting
    manifest = read_json(directory / "manifest.json")
    if not manifest["complete"]:
        raise RuntimeError("Sample generation is incomplete.")
    for record in manifest["records"]:
        if sha256_file(directory / record["file"]) != record["sha256"]:
            raise RuntimeError("A generated sample changed.")
    return directory, manifest, freeze


@_locked
def show_generated_grid(run: Any, analysis: str, setting: str) -> Any:
    directory, manifest, freeze = _sample_records(run, analysis, setting)
    n = min(freeze["evaluation_config"]["grid_per_class"], manifest["identity"]["n_per_class"])
    import random
    selected = []
    for label in range(3):
        group = [row for row in manifest["records"] if row["label"] == label]
        selected.extend(random.Random(freeze["evaluation_config"]["metric_seed"] + label).sample(group, n))
    figure, axes = plt.subplots(3, n, figsize=(max(8, n * 1.35), 4.7), squeeze=False)
    for axis, row in zip(axes.flat, selected):
        with Image.open(directory / row["file"]) as image:
            axis.imshow(np.array(image))
        axis.set_title(f"{run.config.data.display_classes[row['label']]} #{row['index']}", fontsize=7)
        axis.axis("off")
    figure.suptitle(f"{setting} — fixed random grid; assess class adherence qualitatively")
    figure.tight_layout()
    output = _analysis_path(run, analysis) / "figures" / f"{setting}_grid.png"
    _save_figure(figure, output)
    _log_figure(run, freeze["settings"][setting]["family"], f"{analysis}/{setting}", figure)
    return figure


@_locked
def plot_comparison(run: Any, analysis: str, metric: str = "kid") -> Any:
    freeze = load_freeze(run, analysis)
    directory = _analysis_path(run, analysis)
    available = []
    for setting in freeze["settings"]:
        path = directory / "metrics" / f"{setting}.json"
        if path.exists():
            available.append((setting, read_json(path)))
    if not available:
        raise ValueError("No evaluated sample settings exist yet.")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    if metric in ("quality_speed", "speed"):
        points = 0
        for setting, result in available:
            path = directory / "benchmarks" / f"{setting}.json"
            if not path.exists():
                continue
            benchmark = read_json(path)
            latency = benchmark["batches"]["1"]["median_batch_seconds"]
            kid = result["macro_class_kid"]["mean"]
            axis.plot(latency, kid, marker="o")
            axis.annotate(setting, (latency, kid), xytext=(5, 5), textcoords="offset points", fontsize=8)
            points += 1
        if not points:
            plt.close(figure)
            raise ValueError("Run the timing benchmarks before plotting quality versus speed.")
        axis.set(xlabel="Warmed batch-1 median generation seconds (including decoder)", ylabel="Macro-average class KID (lower is better)")
    elif metric in ("kid", "overall_kid", "precision", "recall"):
        names = [name for name, _ in available]
        if metric == "kid":
            axis.bar(names, [row["macro_class_kid"]["mean"] for _, row in available])
            axis.set_ylabel("Macro-average class KID (primary; no inferred uncertainty)")
        elif metric == "overall_kid":
            values = [row["overall"]["kid"]["mean"] for _, row in available]
            errors = [row["overall"]["kid"]["subset_std"] for _, row in available]
            axis.bar(names, values, yerr=errors, capsize=3)
            axis.set_ylabel("Overall KID ± subset standard deviation")
        else:
            axis.bar(names, [row["overall"]["precision_recall"][metric] for _, row in available])
            axis.set_ylabel(f"Inception-space {metric} diagnostic")
            axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)
    else:
        plt.close(figure)
        raise ValueError("metric must be kid, overall_kid, precision, recall, or quality_speed")
    axis.set_title("Observed settings · one training seed · subset variation is not seed uncertainty")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, directory / "figures" / f"comparison_{metric}.png")


@_locked
def plot_generation_process(run: Any, analysis: str = "analysis_001", *, family: str = "diffusion",
                            class_name: str = "cat", seed: int = 6001, steps: int = 50) -> Any:
    load_freeze(run, analysis)
    if family not in ("diffusion", "autoregressive"):
        raise ValueError("family must be diffusion or autoregressive")
    suffix = f"diffusion_{class_name}_{seed}_{steps}" if family == "diffusion" else f"autoregressive_{class_name}_{seed}"
    directory = _analysis_path(run, analysis) / "process" / suffix
    manifest = read_json(directory / "manifest.json")
    rows = manifest["records"]
    figure, axes = plt.subplots(1, len(rows), figsize=(2 * len(rows), 2.8), squeeze=False)
    for axis, row in zip(axes.flat, rows):
        path = directory / row["file"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError("Process illustration changed.")
        with Image.open(path) as image:
            axis.imshow(np.array(image))
        axis.set_title(f"{row.get('snapshot_index', row.get('tokens_visible'))}", fontsize=9)
        axis.axis("off")
    figure.suptitle(manifest["interpretation"], fontsize=9)
    figure.tight_layout()
    return _save_figure(figure, directory / "process.png")


@_locked
def show_nearest_training(run: Any, analysis: str, setting: str, count: int = 6) -> Any:
    """Fixed random generated examples and their nearest training-feature matches."""
    directory, _manifest, freeze = _sample_records(run, analysis, setting)
    records = read_json(_analysis_path(run, analysis) / "nearest_training" / f"{setting}.json")["records"]
    import random
    selected = random.Random(freeze["evaluation_config"]["metric_seed"]).sample(records, min(count, len(records)))
    figure, axes = plt.subplots(2, len(selected), figsize=(2 * len(selected), 4), squeeze=False)
    for col, record in enumerate(selected):
        for row, path in enumerate((directory / record["generated_file"], run.root / record["training_relative_path"])):
            with Image.open(path) as image:
                axes[row, col].imshow(np.array(image.convert("RGB")))
            axes[row, col].axis("off")
        axes[0, col].set_title("Generated", fontsize=9)
        axes[1, col].set_title(f"Nearest train · d={record['inception_distance']:.2f}", fontsize=8)
    figure.suptitle("Feature-neighbor inspection; similarity alone does not establish copying")
    figure.tight_layout()
    return _save_figure(figure, _analysis_path(run, analysis) / "figures" / f"{setting}_nearest.png")


@_locked
def show_ddpm_examples(run: Any, analysis: str, class_name: str = "cat", seed: int = 6001,
                       count: int = 4) -> Any:
    """Display existing ancestral samples without running inference a second time."""
    freeze = load_freeze(run, analysis)
    if class_name not in (*run.config.data.classes, "wildlife") or count < 1:
        raise ValueError("Choose cat, dog or wildlife and a positive count.")
    directory = _analysis_path(run, analysis) / "illustrative" / f"ddpm_{class_name}_{seed}_{count}"
    manifest = read_json(directory / "manifest.json")
    expected = {"freeze_sha256": digest_object(freeze), "class_name": class_name,
                "seed": seed, "count": count, "purpose": "illustrative_only"}
    if manifest["identity"] != expected or len(manifest["records"]) != count:
        raise RuntimeError("Ancestral sample manifest does not match this illustration.")
    figure, axes = plt.subplots(1, count, figsize=(2.5 * count, 3), squeeze=False)
    for axis, row in zip(axes.flat, manifest["records"]):
        path = directory / row["file"]
        if sha256_file(path) != row["sha256"]:
            plt.close(figure)
            raise RuntimeError("An ancestral DDPM illustration changed.")
        with Image.open(path) as image:
            axis.imshow(np.array(image))
        axis.axis("off")
    figure.suptitle(f"Ancestral DDPM · {class_name} · illustrative only (not a quality-study setting)")
    figure.tight_layout()
    _log_figure(run, "diffusion", f"{analysis}/ancestral_ddpm_{class_name}_{seed}", figure)
    return _save_figure(figure, _analysis_path(run, analysis) / "figures" / f"ddpm_{class_name}_{seed}_{count}_grid.png")


@_locked
def write_report(run: Any, analysis: str = "analysis_001") -> Path:
    """Write an honest partial report until both preregistered primary results exist."""
    freeze = load_freeze(run, analysis)
    directory = _analysis_path(run, analysis)
    metrics, benchmarks, artifacts = {}, {}, {}
    for setting in freeze["settings"]:
        for kind, target in (("metrics", metrics), ("benchmarks", benchmarks)):
            path = directory / kind / f"{setting}.json"
            if path.exists():
                target[setting] = read_json(path)
                artifacts[str(path.relative_to(run.path))] = sha256_file(path)
    primary = freeze["primary_settings"]
    complete = all(name in metrics and name in benchmarks for name in primary)
    title = "Animal-face generation: measured comparison" if complete else "Animal-face generation: partial results"
    lines = [f"# {title}", "", f"Run: `{run.path.name}` · evaluation revision: `{analysis}`.", "",
             "DDPM U-Net and causal Transformer priors share one frozen VQ codec on AFHQv2 at 128×128.",
             "The upstream held-out partition serves as the final test set; development validation was carved from training.",
             "Final-epoch priors are compared; diffusion uses EMA. These are observations from one training seed.", ""]
    if all(getattr(run.config, family).training.epochs == 1
           for family in ("tokenizer", "diffusion", "autoregressive")):
        lines.extend([("**One-epoch pipeline test.** Full evaluation exercises the complete workflow; "
                       "these checkpoints do not establish converged image quality or a winning model family."), ""])
    if not complete:
        missing = [f"{name} ({'metrics' if name not in metrics else 'benchmark'})" for name in primary
                   if name not in metrics or name not in benchmarks]
        lines.extend(["**Primary comparison incomplete.** Missing: " + ", ".join(missing) + ". No winner is inferred.", ""])
    lines.extend(["Primary quality metric: macro-average of the three class KIDs. Overall KID is an additional diagnostic; its subset SD is not a macro-KID error bar.", "",
                  "| Setting | Macro class KID (primary) | Overall KID | Overall subset SD | Precision | Recall | Batch-1 median s | Batch-8 images/s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for setting in freeze["settings"]:
        result, benchmark = metrics.get(setting), benchmarks.get(setting)
        if result is None:
            lines.append(f"| {setting} | not evaluated | — | — | — | — | — | — |")
            continue
        kid, pr = result["overall"]["kid"], result["overall"]["precision_recall"]
        latency = f"{benchmark['batches']['1']['median_batch_seconds']:.4f}" if benchmark and "1" in benchmark["batches"] else "—"
        throughput = f"{benchmark['batches']['8']['aggregate_images_per_second']:.3f}" if benchmark and "8" in benchmark["batches"] else "—"
        lines.append(f"| {setting} | {result['macro_class_kid']['mean']:.6f} | {kid['mean']:.6f} | {kid['subset_std']:.6f} | {pr['precision']:.4f} | {pr['recall']:.4f} | {latency} | {throughput} |")
    if complete:
        left, right = primary
        gap = metrics[left]["macro_class_kid"]["mean"] - metrics[right]["macro_class_kid"]["mean"]
        lines.extend(["", f"The observed primary macro-class KID difference ({left} minus {right}) is **{gap:.6f}**.",
                      "This descriptive difference does not establish superiority across seeds, architectures, or datasets."])
    lines.extend(["", "## Training cost and checkpoint artifacts", "",
                  "The codec is a shared training stage, counted once. Equal prior epochs and data exposure do not imply equal parameters, compute, or convergence.", "",
                  "| Stage | Parameters | Epochs | Training image exposures | Train + validation epoch hours | Best MiB | Final MiB | Resume/last MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    data_manifest_path = run.path / "data" / "manifest.json"
    data_manifest = read_json(data_manifest_path)
    if digest_object(data_manifest) != freeze["fixed"]["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed after evaluation freeze.")
    artifacts[str(data_manifest_path.relative_to(run.path))] = sha256_file(data_manifest_path)
    train_count = len(data_manifest["splits"]["train"])
    for family in ("tokenizer", "diffusion", "autoregressive"):
        model_directory = run.model_dir(family)
        architecture_path = model_directory / "architecture.json"
        architecture = read_json(architecture_path)
        history_path = model_directory / "metrics.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if not history:
            raise RuntimeError(f"Cannot report training cost without committed history: {family}")
        final = history[-1]
        epochs = final["epoch"]
        frozen_checkpoint = freeze["fixed"]["checkpoints"][family]
        if (epochs != frozen_checkpoint["epoch"]
                or final["training_seconds"] != frozen_checkpoint["training_seconds"]
                or architecture["total_parameters"] != frozen_checkpoint["parameters"]):
            raise RuntimeError(f"Training-cost artifacts do not match the frozen checkpoint: {family}")
        hours = float(final["training_seconds"]) / 3600
        sizes = {kind: (model_directory / f"{kind}.pt").stat().st_size / 2**20
                 for kind in ("best_validation", "final", "last")}
        label = "VQ codec (shared once)" if family == "tokenizer" else family
        lines.append(f"| {label} | {architecture['total_parameters']:,} | {epochs} | {train_count * epochs:,} | {hours:.4f} | {sizes['best_validation']:.2f} | {sizes['final']:.2f} | {sizes['last']:.2f} |")
        artifacts[str(architecture_path.relative_to(run.path))] = sha256_file(architecture_path)
        artifacts[str(history_path.relative_to(run.path))] = sha256_file(history_path)
    lines.extend(["", "Epoch duration includes training and development validation, and excludes epoch checkpoint serialization. It is measured wall time, not FLOPs or training-only kernel time.",
                  "Concurrent generator stage durations overlap and must not be added to estimate end-to-end wall time. Worker records in jobs/ retain stage timings and telemetry.",
                  "Best, final, and resume/last sizes refer to full saved training-state checkpoints, including optimizer, scheduler, RNG state and EMA where applicable; they are not weight-only sizes. Export files may duplicate the same committed checkpoint and these columns should not be summed as unique storage.", "",
                  "## Sampling latency and observed memory", "",
                  "First call uses a freshly constructed wrapper in the existing process; it is not a fresh-process or cold-OS measurement. Warm timings follow the registered warmups.", "",
                  "| Setting | Batch | First call s | Warm median s | Warm p95 s | Images/s | Max observed RSS MiB | Max observed device allocated MiB | Max observed MPS driver MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    def memory_mib(value: float | None) -> str:
        return "—" if value is None else f"{value / 2**20:.2f}"
    for setting, benchmark in benchmarks.items():
        for batch, record in benchmark["batches"].items():
            maximum = record["memory"]["observed_maxima"]
            allocated = maximum.get("mps_allocated_bytes", maximum.get("cuda_allocated_bytes"))
            lines.append(f"| {setting} | {batch} | {record['first_call_seconds']:.4f} | {record['median_batch_seconds']:.4f} | {record['p95_batch_seconds']:.4f} | {record['aggregate_images_per_second']:.3f} | {memory_mib(maximum.get('process_rss_bytes'))} | {memory_mib(allocated)} | {memory_mib(maximum.get('mps_driver_bytes'))} |")
    for setting in benchmarks:
        lines.append(f"\n[{setting}: raw trials and full memory counter definitions](benchmarks/{setting}.json)")
    lines.extend(["", "RSS, device allocation and MPS driver memory overlap and must not be added. Polling records observed maxima; short-lived peaks may be missed."])
    lines.extend(["", "## Class diagnostics", "",
                  "| Setting | Class | KID | Subset SD | Precision | Recall |",
                  "|---|---|---:|---:|---:|---:|"])
    for setting, result in metrics.items():
        for name, row in result["per_class"].items():
            lines.append(f"| {setting} | {name} | {row['kid']['mean']:.6f} | {row['kid']['subset_std']:.6f} | {row['precision_recall']['precision']:.4f} | {row['precision_recall']['recall']:.4f} |")
    for setting, result in metrics.items():
        lines.append(f"\n{setting}: macro mean class KID = {result['macro_class_kid']['mean']:.6f}; exact duplicate generated PNGs = {result['exact_generated_duplicates']}.")
    reconstruction = run.path / "evaluation" / "reconstructions" / "metrics.json"
    if reconstruction.exists():
        result = read_json(reconstruction)
        lines.extend(["", "## Shared codec", "",
                      (f"Development reconstruction PSNR: {result['psnr']:.3f} dB; LPIPS: {result['lpips']:.5f}; "
                       f"codebook usage: {result['codebook_usage']:.2%}; perplexity: {result['perplexity']:.2f}."),
                      "Codec losses constrain both systems; they are not an absolute mathematical bound on the generation metrics."])
    lines.extend(["", "## Evidence and limits", "",
                  "- KID subset SD measures finite-sample subset variation, not a confidence interval or training-seed uncertainty.",
                  "- Precision/recall uses small-sample Inception neighborhoods and does not certify semantic correctness.",
                  "- Class adherence requires human inspection of fixed random grids; no classifier accuracy is fabricated.",
                  "- Nearest training neighbors help inspect memorization but do not prove copying or its absence.",
                  "- Speed includes shared decoding and CPU uint8 outputs, excludes disk and metrics, and uses explicit synchronization.",
                  "- Memory observations overlap on unified memory and must not be added; sampled MPS maxima are not exact peaks.",
                  "- The next useful action is to inspect class-specific failures and nearest neighbors before interpreting metric differences.", ""])
    figures = sorted((directory / "figures").glob("*.png"))
    missing_figures = [f"{setting}_{kind}.png" for setting in metrics for kind in ("grid", "nearest")
                       if not (directory / "figures" / f"{setting}_{kind}.png").exists()]
    if missing_figures:
        lines.extend(["The following figure cells have not produced artifacts yet: "
                      + ", ".join(f"`{name}`" for name in missing_figures)
                      + ". The report does not generate them automatically.", ""])
    for path in figures:
        lines.extend([f"![{path.stem}](figures/{path.name})", ""])
    lines.extend(["## Research references", "",
                  "- [DDPM](https://arxiv.org/abs/2006.11239)",
                  "- [DDIM](https://arxiv.org/abs/2010.02502)",
                  "- [VQ-VAE](https://arxiv.org/abs/1711.00937)",
                  "- [Taming Transformers](https://arxiv.org/abs/2012.09841)",
                  "- [Latent diffusion models](https://arxiv.org/abs/2112.10752)",
                  "- [Image GPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)",
                  "- [KID](https://arxiv.org/abs/1801.01401)",
                  "- [Improved precision and recall](https://arxiv.org/abs/1904.06991)",
                  "- [AFHQv2 data and noncommercial license](https://github.com/clovaai/stargan-v2)", ""])
    path = directory / "report.md"
    atomic_bytes(path, "\n".join(lines).encode())
    atomic_json(directory / "report_manifest.json", {"primary_complete": complete, "inputs": artifacts,
                                                    "input_paths_relative_to": "run_directory",
                                                    "report_sha256": sha256_file(path)})
    return path
