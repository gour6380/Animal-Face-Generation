"""Explicit acquisition and offline loading of frozen perceptual/evaluation nets."""
from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from .data import _download
from .io import atomic_json, digest_object, immutable_json, read_json, sha256_file
from .runs import Run

VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
INCEPTION_URL = "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"


def acquire_weights(run: Run) -> dict[str, Any]:
    """The only entry point permitted to fetch pretrained weight files."""
    with run.lock(), FileLock(run.root / "data" / ".acquisition.lock", timeout=0):
        run.verify()
        directory = run.root / "data" / "weights"
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            records = {}
            for name, url in (("vgg16", VGG_URL), ("inception", INCEPTION_URL)):
                path = directory / f"{name}.pt"
                _download(url, path)
                actual = sha256_file(path)
                if name == "vgg16" and not actual.startswith("397923af"):
                    raise ValueError(f"Publisher filename checksum prefix mismatch for {name}.")
                records[name] = {"path": str(path.relative_to(run.root)), "url": url, "sha256": actual, "bytes": path.stat().st_size,
                    "checksum_basis": "PyTorch filename SHA256 prefix plus observed full SHA256" if name == "vgg16" else "Observed SHA256; filename suffix is not assumed to be a published SHA256"}
            installed = Path(importlib.metadata.distribution("lpips").locate_file("lpips/weights/v0.1/vgg.pth"))
            if not installed.is_file():
                raise FileNotFoundError("Pinned lpips distribution lacks v0.1 VGG calibration weights.")
            calibration = directory / "lpips_vgg_v0.1.pt"
            shutil.copyfile(installed, calibration)
            records["lpips"] = {"path": str(calibration.relative_to(run.root)), "sha256": sha256_file(calibration),
                "source": "installed lpips/weights/v0.1/vgg.pth", "package_version": importlib.metadata.version("lpips"), "bytes": calibration.stat().st_size}
            manifest = {"schema_version": 1, "weights": records,
                "purpose": "Frozen pretrained perceptual supervision and Inception evaluation; generator/codec parameters initialize from scratch."}
            manifest["digest"] = digest_object(manifest)
            atomic_json(manifest_path, manifest)
        for record in manifest["weights"].values():
            if sha256_file(run.root / record["path"]) != record["sha256"]:
                raise ValueError("Cached feature-network weights changed.")
        immutable_json(run.path / "data" / "weights.json", manifest)
        return manifest


def _weight_paths(run: Run) -> dict[str, Path]:
    receipt = run.path / "data" / "weights.json"
    if not receipt.exists():
        raise FileNotFoundError("Run the explicit acquire_weights notebook cell first; loaders never download.")
    manifest = read_json(receipt)
    paths = {}
    for name, record in manifest["weights"].items():
        path = run.root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Feature-network weight identity changed: {name}")
        paths[name] = path
    return paths


def load_lpips(run: Run, device: str | torch.device) -> torch.nn.Module:
    import lpips
    paths = _weight_paths(run)
    # Both switches disable constructor downloads. Load each weight set explicitly.
    metric = lpips.LPIPS(net="vgg", version="0.1", pretrained=False, pnet_rand=True, verbose=False)
    backbone = torch.load(paths["vgg16"], map_location="cpu", weights_only=True)
    net_state = {key: backbone["features." + key.split(".", 1)[1]] for key in metric.net.state_dict()}
    metric.net.load_state_dict(net_state, strict=True)
    del backbone, net_state
    calibration = torch.load(paths["lpips"], map_location="cpu", weights_only=True)
    result = metric.load_state_dict(calibration, strict=False)
    # LPIPS registers the same calibration modules as lin0..lin4 AND in lins;
    # published weights use the former names, so missing aliases are expected.
    complete_linear_weights = all(f"lin{i}.model.1.weight" in calibration for i in range(5))
    if result.unexpected_keys or not complete_linear_weights:
        raise ValueError("LPIPS calibration weights do not match the pinned implementation.")
    metric.requires_grad_(False)
    metric.eval()
    return metric.to(device)


def load_inception(run: Run, device: str | torch.device) -> torch.nn.Module:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    paths = _weight_paths(run)
    model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_weights_path=str(paths["inception"]), feature_extractor_internal_dtype="float32")
    return model.eval().requires_grad_(False).to(device)
