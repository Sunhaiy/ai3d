from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tiny3d.image_data import (
    find_images,
    load_square_image,
    render_voxel_projection,
    save_image_voxel_dataset,
)
from tiny3d.mesh import voxels_to_obj
from tiny3d.mesh_data import find_meshes, load_mesh, mesh_to_voxels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired image-to-3D training data.")
    parser.add_argument("--meshes", required=True, help="Directory of target 3D mesh files.")
    parser.add_argument(
        "--images",
        help="Optional directory of matching images. Names must match mesh names.",
    )
    parser.add_argument("--output", default="data/image3d_pairs.npz")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--views",
        type=int,
        default=1,
        help="Synthetic yaw views per mesh when --images is omitted.",
    )
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--preview-dir", default="data/image3d_previews")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def yaw_matrix(angle: float) -> np.ndarray:
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


def main() -> None:
    args = parse_args()
    if args.views < 1:
        raise ValueError("views must be at least 1")
    if args.preview_count < 0:
        raise ValueError("preview-count cannot be negative")
    if args.images and args.views != 1:
        raise ValueError("--views is only used for automatically rendered images")

    mesh_paths = find_meshes(args.meshes)
    if not mesh_paths:
        raise RuntimeError("no supported mesh files were found")
    image_index = find_images(args.images) if args.images else None
    all_images: list[np.ndarray] = []
    all_voxels: list[np.ndarray] = []
    voxel_indices: list[int] = []
    sources: list[str] = []
    failed = 0

    for mesh_index, mesh_path in enumerate(mesh_paths, start=1):
        try:
            mesh = load_mesh(mesh_path)
            if image_index is not None:
                matching_images = image_index.get(mesh_path.stem.casefold(), [])
                if not matching_images:
                    raise ValueError("no image with the same base name was found")
                voxels = mesh_to_voxels(mesh, args.resolution, fill=not args.no_fill)
                voxel_index = len(all_voxels)
                all_voxels.append(voxels)
                for image_path in matching_images:
                    all_images.append(load_square_image(image_path, args.image_size))
                    voxel_indices.append(voxel_index)
                    sources.append(f"{image_path}|{mesh_path}")
            else:
                for view_index in range(args.views):
                    angle = 2.0 * np.pi * view_index / args.views
                    voxels = mesh_to_voxels(
                        mesh,
                        args.resolution,
                        fill=not args.no_fill,
                        transform=yaw_matrix(angle),
                    )
                    all_images.append(render_voxel_projection(voxels, args.image_size))
                    all_voxels.append(voxels)
                    voxel_indices.append(len(all_voxels) - 1)
                    sources.append(f"auto-view-{view_index}|{mesh_path}")
            print(f"[{mesh_index}/{len(mesh_paths)}] OK   {mesh_path}")
        except Exception as error:
            failed += 1
            print(f"[{mesh_index}/{len(mesh_paths)}] SKIP {mesh_path}: {error}", file=sys.stderr)
            if args.fail_fast:
                raise

    if not all_images:
        raise RuntimeError("no valid image and mesh pairs were created")

    image_array = np.stack(all_images)
    voxel_array = np.stack(all_voxels)
    voxel_index_array = np.asarray(voxel_indices, dtype=np.int32)
    save_image_voxel_dataset(
        args.output,
        image_array,
        voxel_array,
        sources=sources,
        voxel_indices=voxel_index_array,
    )

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(args.preview_count, len(image_array))):
        Image.fromarray(image_array[index], mode="RGB").save(
            preview_dir / f"preview_{index:04d}_input.png"
        )
        voxels_to_obj(
            voxel_array[voxel_index_array[index]],
            preview_dir / f"preview_{index:04d}_target.obj",
        )

    print(f"Saved {len(image_array)} image/3D pairs to {args.output}")
    print(
        f"Images: {image_array.shape}; unique voxel targets: {voxel_array.shape}; "
        f"indices: {voxel_index_array.shape}"
    )
    print(f"Preview pairs: {min(args.preview_count, len(image_array))} in {preview_dir}")
    if failed:
        print(f"Skipped {failed} meshes; review the SKIP lines above.")


if __name__ == "__main__":
    main()
