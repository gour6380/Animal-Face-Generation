"""Checkpoint-only image generation and immutable, balanced sampling artifacts."""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .io import atomic_bytes, atomic_json, digest_object, read_json, sha256_file


def setting_spec(family: str, setting: str) -> dict[str, Any]:
    """Parse one named setting; explicit families prevent mislabeled outputs."""
    if family == "diffusion" and setting.startswith("ddim_"):
        steps = int(setting.removeprefix("ddim_"))
        if steps < 2:
            raise ValueError("DDIM requires at least two steps.")
        return {"family": family, "sampler": "ddim", "steps": steps, "eta": 0.0}
    if family == "autoregressive" and setting.startswith("temperature_"):
        temperature = float(setting.removeprefix("temperature_"))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite.")
        return {"family": family, "temperature": temperature, "use_cache": True,
                "top_k": None, "top_p": None}
    raise ValueError(f"Invalid {family!r} sampling setting: {setting!r}")


def image_to_uint8(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def verify_codec_metadata(metadata: dict[str, Any], codec: dict[str, Any]) -> None:
    for key in ("codec_sha256", "mean", "std", "token_cache_identity"):
        if key not in codec or metadata.get(key) != codec[key]:
            raise RuntimeError(f"Prior checkpoint does not match the frozen codec: {key}")


def _atomic_png(path: Path, tensor: torch.Tensor) -> str:
    """Hash the encoded bytes once; only reread a preexisting file on recovery."""
    array = tensor.permute(1, 2, 0).contiguous().numpy()
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    content = stream.getvalue()
    checksum = hashlib.sha256(content).hexdigest()
    if path.exists():
        if sha256_file(path) != checksum:
            raise RuntimeError(f"Existing sample differs: {path}. Use a new analysis revision.")
    else:
        atomic_bytes(path, content)
    return checksum


class Generator:
    """An inference wrapper requiring configs, checkpoints and codec metadata only."""

    def __init__(self, run: Any, family: str):
        from .checkpoints import load_model_checkpoint
        from .models.autoregressive import ARTransformer
        from .models.diffusion import DiffusionSchedule, DiffusionUNet
        from .tokenization import load_frozen_tokenizer

        if family not in ("diffusion", "autoregressive"):
            raise ValueError("family must be diffusion or autoregressive")
        self.run, self.family, self.device = run, family, torch.device(run.device)
        self.codec, self.codec_manifest = load_frozen_tokenizer(run, str(self.device))
        self.codec.eval().requires_grad_(False)
        cfg = run.config
        checkpoint = load_model_checkpoint(run, family, kind="final")
        verify_codec_metadata(checkpoint["metadata"], self.codec_manifest)
        if family == "diffusion":
            self.model = DiffusionUNet(cfg.diffusion, latent_channels=cfg.tokenizer.embedding_dim,
                                       num_classes=3, latent_size=cfg.latent_size).to(self.device)
            state = checkpoint.get("ema")
            if state is None:
                raise RuntimeError("Diffusion final checkpoint is missing EMA weights.")
            self.model.load_state_dict(state)
            self.schedule = DiffusionSchedule(cfg.diffusion).to(self.device)
            self.mean, self.std = float(self.codec_manifest["mean"]), float(self.codec_manifest["std"])
            if not np.isfinite(self.std) or self.std <= 0:
                raise ValueError("Invalid frozen latent standard deviation.")
        else:
            self.model = ARTransformer(cfg.autoregressive,
                                       codebook_size=cfg.tokenizer.codebook_size,
                                       sequence_length=cfg.latent_size ** 2,
                                       num_classes=3).to(self.device)
            self.model.load_state_dict(checkpoint["model"])
        self.model.eval().requires_grad_(False)

    def label(self, class_name: str) -> int:
        class_name = "wild" if class_name == "wildlife" else class_name
        try:
            return self.run.config.data.classes.index(class_name)
        except ValueError as exc:
            raise ValueError("class_name must be cat, dog, or wildlife") from exc

    @torch.inference_mode()
    def generate(self, class_name: str, count: int = 1, seed: int = 6001,
                 *, steps: int = 50, temperature: float = 1.0,
                 sampler: str = "ddim") -> torch.Tensor:
        """Return BCHW uint8 CPU images; no raw images or token caches are opened."""
        if count < 1:
            raise ValueError("count must be positive")
        labels = torch.full((count,), self.label(class_name), dtype=torch.long, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if self.family == "autoregressive":
            ids = self.model.generate(labels, generator=generator, temperature=temperature, use_cache=True)
        else:
            shape = (count, self.run.config.tokenizer.embedding_dim,
                     self.run.config.latent_size, self.run.config.latent_size)
            noise = torch.randn(shape, generator=generator, device="cpu").to(self.device)
            if sampler == "ddim":
                latent = self.schedule.ddim_sample(self.model, noise, labels, steps=steps)
            elif sampler == "ddpm":
                latent = self.schedule.ddpm_sample(self.model, noise, labels, generator=generator)
            else:
                raise ValueError("sampler must be ddim or ddpm")
            ids = self.codec.nearest_codes(latent * self.std + self.mean)
        return image_to_uint8(self.codec.decode(self.codec.lookup(ids)))


def load_generator(run: Any, family: str) -> Generator:
    run.verify()
    return Generator(run, family)


def generate_samples(run: Any, family: str, setting: str,
                     analysis: str = "analysis_001") -> dict[str, Any]:
    """Generate balanced samples. Interrupted batches restart with recorded batch seeds."""
    from .evaluate import _prepare_reference, load_freeze

    freeze = load_freeze(run, analysis)
    spec = setting_spec(family, setting)
    if setting not in freeze["settings"] or freeze["settings"][setting] != spec:
        raise ValueError("Sampling setting was not frozen in this analysis revision.")
    reference = _prepare_reference(run, analysis, freeze)
    config = freeze["evaluation_config"]
    directory = run.path / "evaluation" / analysis / "samples" / setting
    manifest_path = directory / "manifest.json"
    identity = {"freeze_sha256": digest_object(freeze), "setting": spec,
                "n_per_class": reference["n_per_class"], "seed": config["sampling_seed"],
                "batch_size": config["generation_batch"]}
    with run.lock():
        directory.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity"] != identity:
                raise RuntimeError("Sampling manifest mismatch; start a new analysis revision.")
            for record in manifest["records"]:
                if sha256_file(directory / record["file"]) != record["sha256"]:
                    raise RuntimeError(f"Sample changed: {record['file']}")
            if manifest.get("complete"):
                return manifest
        else:
            manifest = {"identity": identity, "records": [], "complete": False}
            atomic_json(manifest_path, manifest)
        wrapper = load_generator(run, family)
        completed = {row["file"] for row in manifest["records"]}
        batch_size, n = config["generation_batch"], reference["n_per_class"]
        pending: list[tuple[Future[str], dict[str, Any]]] = []

        def commit_pending() -> None:
            # Resolve in manifest order, independently of write completion order.
            for future, record in pending:
                checksum = future.result()  # Surface all encoder/write failures.
                if record["file"] not in completed:
                    manifest["records"].append({**record, "sha256": checksum})
                    completed.add(record["file"])
            manifest["records"].sort(key=lambda row: (row["label"], row["index"]))
            if pending:
                atomic_json(manifest_path, manifest)
            pending.clear()

        # At most one previous batch is being encoded/written while the next
        # batch runs on the accelerator. Keep CPU memory and network I/O bounded.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="sample-png") as writers:
            for label, class_name in enumerate(run.config.data.classes):
                for start in tqdm(range(0, n, batch_size), desc=f"{setting}: {class_name}", unit="batch"):
                    count = min(batch_size, n - start)
                    filenames = [f"{class_name}/{i:05d}.png" for i in range(start, start + count)]
                    if all(name in completed for name in filenames):
                        continue
                    seed = config["sampling_seed"] + label * 100_000 + start
                    images = wrapper.generate(class_name, count, seed,
                                              steps=spec.get("steps", 50),
                                              temperature=spec.get("temperature", 1.0))
                    commit_pending()
                    for offset, filename in enumerate(filenames):
                        record = {"file": filename, "label": label, "class_name": class_name,
                                  "index": start + offset, "batch_seed": seed}
                        pending.append((writers.submit(_atomic_png, directory / filename, images[offset]), record))
            commit_pending()
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        return manifest


def generate_ddpm_examples(run: Any, class_name: str = "cat", count: int = 4,
                           seed: int = 6001, analysis: str = "analysis_001") -> dict[str, Any]:
    """Illustrative ancestral sampling, excluded from the six-setting quality study."""
    from .evaluate import load_freeze
    freeze = load_freeze(run, analysis)
    directory = run.path / "evaluation" / analysis / "illustrative" / f"ddpm_{class_name}_{seed}_{count}"
    identity = {"freeze_sha256": digest_object(freeze), "class_name": class_name,
                "seed": seed, "count": count, "purpose": "illustrative_only"}
    with run.lock():
        path = directory / "manifest.json"
        if path.exists():
            existing = read_json(path)
            if existing["identity"] != identity:
                raise RuntimeError("Illustration identity changed.")
            for row in existing["records"]:
                if sha256_file(directory / row["file"]) != row["sha256"]:
                    raise RuntimeError("Illustration changed.")
            return existing
        wrapper = load_generator(run, "diffusion")
        images = wrapper.generate(class_name, count, seed, sampler="ddpm")
        records = [{"file": f"{i:03d}.png", "sha256": _atomic_png(directory / f"{i:03d}.png", item)}
                   for i, item in enumerate(images)]
        result = {"identity": identity, "records": records}
        atomic_json(path, result)
        return result


@torch.inference_mode()
def generate_diffusion_process(run: Any, class_name: str = "cat", seed: int = 6001,
                               steps: int = 50, analysis: str = "analysis_001") -> Path:
    from .evaluate import load_freeze
    freeze = load_freeze(run, analysis)
    directory = run.path / "evaluation" / analysis / "process" / f"diffusion_{class_name}_{seed}_{steps}"
    path = directory / "manifest.json"
    identity = {"freeze_sha256": digest_object(freeze), "class_name": class_name,
                "seed": seed, "steps": steps, "purpose": "illustrative_only"}
    with run.lock():
        if path.exists():
            old = read_json(path)
            if old["identity"] != identity or any(
                    sha256_file(directory / r["file"]) != r["sha256"] for r in old["records"]):
                raise RuntimeError("Process illustration changed.")
            return path
        wrapper = load_generator(run, "diffusion")
        labels = torch.tensor([wrapper.label(class_name)], device=wrapper.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, run.config.tokenizer.embedding_dim, run.config.latent_size,
                             run.config.latent_size), generator=generator).to(wrapper.device)
        _, snapshots = wrapper.schedule.ddim_sample(wrapper.model, noise, labels, steps=steps, trajectory=True)
        # Eight evenly spaced snapshots keep artifacts small; these are decoded noisy states.
        selected = sorted(set(np.linspace(0, len(snapshots) - 1, min(8, len(snapshots)), dtype=int).tolist()))
        records = []
        for index in selected:
            latent = snapshots[index].to(wrapper.device) * wrapper.std + wrapper.mean
            decoded = image_to_uint8(wrapper.codec.decode(wrapper.codec.lookup(wrapper.codec.nearest_codes(latent))))
            filename = f"state_{index:04d}.png"
            records.append({"file": filename, "snapshot_index": index,
                            "sha256": _atomic_png(directory / filename, decoded[0])})
        atomic_json(path, {"identity": identity, "records": records,
                           "interpretation": "Decoded, snapped intermediate noisy latent states; not calibrated images."})
        return path


@torch.inference_mode()
def generate_autoregressive_process(run: Any, class_name: str = "cat", seed: int = 6001,
                                    analysis: str = "analysis_001") -> Path:
    """Show actual sampled token prefixes, never decode invented future tokens."""
    from matplotlib import colormaps

    from .evaluate import load_freeze
    freeze = load_freeze(run, analysis)
    directory = run.path / "evaluation" / analysis / "process" / f"autoregressive_{class_name}_{seed}"
    path = directory / "manifest.json"
    identity = {"freeze_sha256": digest_object(freeze), "class_name": class_name,
                "seed": seed, "temperature": 1.0, "purpose": "illustrative_only"}
    with run.lock():
        if path.exists():
            previous = read_json(path)
            if previous["identity"] != identity or any(
                    sha256_file(directory / row["file"]) != row["sha256"] for row in previous["records"]):
                raise RuntimeError("Autoregressive process artifact changed.")
            return path
        wrapper = load_generator(run, "autoregressive")
        labels = torch.tensor([wrapper.label(class_name)], device=wrapper.device)
        ids = wrapper.model.generate(labels, generator=torch.Generator(device="cpu").manual_seed(seed),
                                     temperature=1.0, use_cache=True).cpu()[0]
        flat = ids.numpy().reshape(-1)
        colors = (colormaps["turbo"](flat / (run.config.tokenizer.codebook_size - 1))[:, :3] * 255).round().astype(np.uint8)
        records = []
        for count in (0, 64, 128, 192, 256):
            canvas = np.full_like(colors, 24)
            canvas[:count] = colors[:count]
            canvas = canvas.reshape(run.config.latent_size, run.config.latent_size, 3)
            enlarged = np.repeat(np.repeat(canvas, 8, axis=0), 8, axis=1)
            filename = f"tokens_{count:03d}.png"
            records.append({"file": filename, "tokens_visible": count,
                            "sha256": _atomic_png(directory / filename, torch.from_numpy(enlarged).permute(2, 0, 1))})
        atomic_json(path, {"identity": identity, "records": records, "token_ids": flat.tolist(),
                           "interpretation": "Actual raster-order token prefixes; colors encode token IDs, dark cells are ungenerated (not RGB previews)."})
        return path
