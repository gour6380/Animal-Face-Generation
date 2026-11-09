"""Packed, immutable RGB snapshots with a content hash for every image row."""
from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
U = TypeVar("U")
SNAPSHOT_VERSION = 1


def ordered_parallel_map(function: Callable[[T], U], values: Iterable[T], workers: int) -> Iterator[U]:
    """Bound in-flight decoded images, retain deterministic source ordering."""
    source = iter(values)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(2 * workers):
            try:
                pending.append(pool.submit(function, next(source)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(function, next(source)))
            except StopIteration:
                pass


def open_snapshot(root: Path, metadata: dict[str, Any]) -> np.memmap:
    """Validate the container header; image bytes are verified when consumed."""
    if metadata["version"] != SNAPSHOT_VERSION or metadata["dtype"] != "uint8":
        raise ValueError("Unsupported preprocessed snapshot version or dtype.")
    path = root / metadata["path"]
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (not isinstance(array, np.memmap) or array.dtype != np.uint8
            or list(array.shape) != metadata["shape"] or array.flags.f_contiguous
            or array.offset + array.nbytes != path.stat().st_size):
        raise ValueError("Preprocessed snapshot shape, layout, or file size changed.")
    return array


def verified_snapshot_row(array: np.ndarray, index: int, checksum: str, image_id: str) -> np.ndarray:
    """Copy before hashing so verification and consumption use identical bytes."""
    image = np.array(array[index], dtype=np.uint8, copy=True)
    if hashlib.sha256(image.tobytes()).hexdigest() != checksum:
        raise ValueError(f"Preprocessed image bytes changed: {image_id}")
    return image
