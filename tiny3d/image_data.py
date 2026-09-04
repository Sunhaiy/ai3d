from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_square_image(path: str | Path, image_size: int) -> np.ndarray:
    if image_size < 32 or image_size % 16 != 0:
        raise ValueError("image_size must be at least 32 and divisible by 16")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.contain(
            image,
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
        offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
        canvas.paste(image, offset)
        return np.asarray(canvas, dtype=np.uint8)


def render_voxel_projection(voxels: np.ndarray, image_size: int) -> np.ndarray:
    if voxels.ndim != 3 or len(set(voxels.shape)) != 1:
        raise ValueError("expected a cubic 3D voxel grid")
    occupied = voxels.astype(bool)
    silhouette = occupied.any(axis=1)
    first_hit = np.argmax(occupied, axis=1)
    thickness = occupied.sum(axis=1)
    resolution = occupied.shape[0]

    depth_light = 1.0 - first_hit / max(1, resolution - 1)
    thickness_light = thickness / max(1, resolution)
    intensity = (0.55 + 0.3 * depth_light + 0.15 * thickness_light) * silhouette
    image = np.clip(intensity.T[::-1] * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    rendered = Image.fromarray(rgb, mode="RGB").resize(
        (image_size, image_size),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(rendered, dtype=np.uint8)


def find_images(directory: str | Path) -> dict[str, list[Path]]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        base_name = path.stem.split("__", 1)[0].casefold()
        index.setdefault(base_name, []).append(path)
    return index


def save_image_voxel_dataset(
    path: str | Path,
    images: np.ndarray,
    voxels: np.ndarray,
    *,
    sources: list[str],
    voxel_indices: np.ndarray | None = None,
) -> None:
    if len(images) != len(sources):
        raise ValueError("images and sources must have equal lengths")
    if voxel_indices is None:
        if len(images) != len(voxels):
            raise ValueError("images and voxels must have equal lengths without voxel indices")
        voxel_indices = np.arange(len(images), dtype=np.int32)
    else:
        voxel_indices = np.asarray(voxel_indices, dtype=np.int32)
        if len(voxel_indices) != len(images):
            raise ValueError("voxel indices and images must have equal lengths")
        if (
            len(voxels) == 0
            or voxel_indices.min(initial=0) < 0
            or voxel_indices.max(initial=0) >= len(voxels)
        ):
            raise ValueError("voxel index is outside the voxel target array")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        images=images.astype(np.uint8, copy=False),
        voxels=voxels.astype(np.uint8, copy=False),
        voxel_indices=voxel_indices,
        sources=np.asarray(sources),
        image_size=np.asarray(images.shape[1], dtype=np.int32),
        resolution=np.asarray(voxels.shape[1], dtype=np.int32),
        pair_count=np.asarray(len(images), dtype=np.int32),
        target_count=np.asarray(len(voxels), dtype=np.int32),
    )


class ImageVoxelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path):
        with np.load(path) as archive:
            images = archive["images"]
            voxels = archive["voxels"]
            voxel_indices = (
                archive["voxel_indices"]
                if "voxel_indices" in archive.files
                else np.arange(len(images), dtype=np.int32)
            )
            sources = archive["sources"] if "sources" in archive.files else None
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("expected images shaped [N, height, width, 3]")
        if voxels.ndim != 4 or len(set(voxels.shape[1:])) != 1:
            raise ValueError("expected voxels shaped [N, resolution, resolution, resolution]")
        if len(voxel_indices) != len(images):
            raise ValueError("image and voxel index counts do not match")
        if (
            len(voxels) == 0
            or voxel_indices.min(initial=0) < 0
            or voxel_indices.max(initial=0) >= len(voxels)
        ):
            raise ValueError("voxel index is outside the voxel target array")
        if images.shape[1] != images.shape[2]:
            raise ValueError("training images must be square")

        self.images = (
            torch.from_numpy(images.astype(np.uint8, copy=False))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.voxels = torch.from_numpy(voxels.astype(np.uint8, copy=False))
        self.voxel_indices = torch.from_numpy(voxel_indices.astype(np.int64, copy=False))
        self.sources = [str(source) for source in sources] if sources is not None else None

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].float().div_(255.0)
        voxel = self.voxels[self.voxel_indices[index]].float().unsqueeze(0)
        return image, voxel
