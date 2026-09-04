from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".3mf"}


def find_meshes(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"mesh directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("file did not contain a mesh")
    if loaded.vertices.shape[0] < 4 or loaded.faces.shape[0] < 4:
        raise ValueError("mesh has too few vertices or faces")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("mesh contains non-finite vertex coordinates")
    return loaded


def random_rotation(mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "none":
        return np.eye(4, dtype=np.float64)
    if mode == "yaw":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array(
            [
                [cosine, -sine, 0.0, 0.0],
                [sine, cosine, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if mode == "all":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=3)
        return trimesh.transformations.euler_matrix(*angles, axes="sxyz")
    raise ValueError(f"unknown rotation mode: {mode}")


def mesh_to_voxels(
    mesh: trimesh.Trimesh,
    resolution: int,
    *,
    fill: bool = True,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    if resolution < 8 or resolution % 8 != 0:
        raise ValueError("resolution must be at least 8 and divisible by 8")

    normalized = mesh.copy()
    if transform is not None:
        normalized.apply_transform(transform)
    bounds = normalized.bounds
    center = bounds.mean(axis=0)
    max_extent = float((bounds[1] - bounds[0]).max())
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("mesh has zero or invalid extent")

    normalized.apply_translation(-center)
    normalized.apply_scale(1.7 / max_extent)
    pitch = 2.0 / resolution
    voxel_grid = normalized.voxelized(pitch=pitch)
    if fill:
        voxel_grid = voxel_grid.fill()

    points = np.asarray(voxel_grid.points)
    if points.size == 0:
        raise ValueError("voxelization produced an empty grid")
    indices = np.floor((points + 1.0) / pitch).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < resolution), axis=1)
    indices = indices[valid]
    if len(indices) == 0:
        raise ValueError("all voxels fell outside the normalized grid")

    voxels = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return voxels
