from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _coordinates(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _primitive(
    rng: np.random.Generator,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    x, y, z = coords
    center = rng.uniform(-0.3, 0.3, size=3)
    kind = rng.choice(("ellipsoid", "box", "cylinder"), p=(0.5, 0.3, 0.2))

    if kind == "ellipsoid":
        radii = rng.uniform(0.22, 0.62, size=3)
        return (
            ((x - center[0]) / radii[0]) ** 2
            + ((y - center[1]) / radii[1]) ** 2
            + ((z - center[2]) / radii[2]) ** 2
            <= 1.0
        )

    if kind == "box":
        half_size = rng.uniform(0.18, 0.52, size=3)
        return (
            (np.abs(x - center[0]) <= half_size[0])
            & (np.abs(y - center[1]) <= half_size[1])
            & (np.abs(z - center[2]) <= half_size[2])
        )

    axis = int(rng.integers(0, 3))
    radius = float(rng.uniform(0.2, 0.48))
    half_length = float(rng.uniform(0.25, 0.65))
    shifted = (x - center[0], y - center[1], z - center[2])
    longitudinal = shifted[axis]
    radial = shifted[(axis + 1) % 3] ** 2 + shifted[(axis + 2) % 3] ** 2
    return (np.abs(longitudinal) <= half_length) & (radial <= radius**2)


def make_shape(
    rng: np.random.Generator,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    shape = _primitive(rng, coords)
    for _ in range(int(rng.integers(0, 3))):
        shape |= _primitive(rng, coords)
    return shape.astype(np.uint8)


def generate_dataset(samples: int, resolution: int, seed: int) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if resolution < 8 or resolution % 8 != 0:
        raise ValueError("resolution must be at least 8 and divisible by 8")

    rng = np.random.default_rng(seed)
    coords = _coordinates(resolution)
    voxels = np.empty((samples, resolution, resolution, resolution), dtype=np.uint8)
    for index in range(samples):
        voxels[index] = make_shape(rng, coords)
    return voxels


def save_dataset(
    path: str | Path,
    voxels: np.ndarray,
    seed: int,
    *,
    sources: list[str] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "voxels": voxels,
        "seed": np.asarray(seed, dtype=np.int64),
    }
    if sources is not None:
        payload["sources"] = np.asarray(sources)
    np.savez_compressed(destination, **payload)


class VoxelDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str | Path):
        with np.load(path) as archive:
            voxels = archive["voxels"]
        if voxels.ndim != 4 or len(set(voxels.shape[1:])) != 1:
            raise ValueError("expected voxels shaped [N, resolution, resolution, resolution]")
        self.voxels = torch.from_numpy(voxels.astype(np.float32, copy=False)).unsqueeze(1)

    def __len__(self) -> int:
        return self.voxels.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.voxels[index]
