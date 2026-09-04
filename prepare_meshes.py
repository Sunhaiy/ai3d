from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tiny3d.data import save_dataset
from tiny3d.mesh import voxels_to_obj
from tiny3d.mesh_data import find_meshes, load_mesh, mesh_to_voxels, random_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a directory of 3D meshes to voxel training data.")
    parser.add_argument("--input", required=True, help="Directory containing OBJ/STL/PLY/GLB/GLTF files.")
    parser.add_argument("--output", default="data/my_shapes.npz")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--augment", type=int, default=1, help="Samples to create per source mesh.")
    parser.add_argument("--rotation", choices=("none", "yaw", "all"), default="yaw")
    parser.add_argument("--no-fill", action="store_true", help="Keep only mesh surface voxels.")
    parser.add_argument("--preview-dir", default="data/previews")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.augment < 1:
        raise ValueError("augment must be at least 1")
    if args.preview_count < 0:
        raise ValueError("preview-count cannot be negative")

    paths = find_meshes(args.input)
    if not paths:
        raise RuntimeError("no supported mesh files were found")
    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    sources: list[str] = []
    failures: list[tuple[Path, str]] = []

    for mesh_index, path in enumerate(paths, start=1):
        try:
            mesh = load_mesh(path)
            for copy_index in range(args.augment):
                transform = None if copy_index == 0 else random_rotation(args.rotation, rng)
                voxels = mesh_to_voxels(
                    mesh,
                    args.resolution,
                    fill=not args.no_fill,
                    transform=transform,
                )
                samples.append(voxels)
                sources.append(f"{path}#{copy_index}")
            print(f"[{mesh_index}/{len(paths)}] OK   {path}")
        except Exception as error:
            failures.append((path, str(error)))
            print(f"[{mesh_index}/{len(paths)}] SKIP {path}: {error}", file=sys.stderr)
            if args.fail_fast:
                raise

    if not samples:
        raise RuntimeError("none of the mesh files could be converted")

    array = np.stack(samples)
    save_dataset(args.output, array, args.seed, sources=sources)
    preview_dir = Path(args.preview_dir)
    for index, voxels in enumerate(array[: args.preview_count]):
        voxels_to_obj(voxels, preview_dir / f"preview_{index:04d}.obj")

    occupancy = 100.0 * float(array.mean())
    print(f"Saved {len(array)} samples from {len(paths) - len(failures)} meshes to {args.output}")
    print(f"Shape: {array.shape}; mean occupancy: {occupancy:.2f}%")
    print(f"Preview OBJ files: {min(args.preview_count, len(array))} in {preview_dir}")
    if failures:
        print(f"Skipped {len(failures)} unreadable meshes; review the SKIP lines above.")


if __name__ == "__main__":
    main()

