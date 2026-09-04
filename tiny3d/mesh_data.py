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


def normalize_mesh(
    mesh: trimesh.Trimesh,
    transform: np.ndarray | None = None,
) -> trimesh.Trimesh:
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
    return normalized


def sample_implicit_occupancy(
    mesh: trimesh.Trimesh,
    resolution: int,
    point_count: int,
    *,
    seed: int,
    transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if resolution < 512 or resolution & (resolution - 1):
        raise ValueError("implicit resolution must be a power of two of at least 512")
    if point_count < 4096:
        raise ValueError("point_count must be at least 4096")

    normalized = normalize_mesh(mesh, transform)
    if not normalized.is_watertight:
        raise ValueError("implicit occupancy sampling requires a watertight mesh")
    rng = np.random.default_rng(seed)
    uniform_count = point_count // 2
    surface_count = point_count - uniform_count
    uniform = rng.uniform(-1.0, 1.0, size=(uniform_count, 3))
    surface, face_indices = trimesh.sample.sample_surface(
        normalized,
        surface_count,
        seed=rng,
    )
    normal_offsets = rng.normal(
        0.0,
        4.0 / resolution,
        size=(surface_count, 1),
    )
    near_surface = surface + normalized.face_normals[face_indices] * normal_offsets
    points = np.concatenate((uniform, near_surface), axis=0)
    points = np.clip(points, -1.0, 1.0)
    grid_indices = np.rint((points + 1.0) * 0.5 * (resolution - 1))
    points = grid_indices / (resolution - 1) * 2.0 - 1.0
    occupancies = normalized.contains(points)
    if not occupancies.any() or occupancies.all():
        raise ValueError(
            "implicit occupancy sampling needs a closed mesh with inside and outside points"
        )
    return points.astype(np.float16), occupancies.astype(np.uint8)


def mesh_to_voxels(
    mesh: trimesh.Trimesh,
    resolution: int,
    *,
    fill: bool = True,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    if resolution < 8 or resolution % 8 != 0:
        raise ValueError("resolution must be at least 8 and divisible by 8")

    normalized = normalize_mesh(mesh, transform)
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
