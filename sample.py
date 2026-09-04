from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tiny3d.mesh import binarize_voxels, voxels_to_obj
from tiny3d.model import VoxelVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample OBJ files from a trained voxel VAE.")
    parser.add_argument("--checkpoint", default="runs/tiny3d.pt")
    parser.add_argument("--output", default="samples")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be at least 1")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = VoxelVAE(
        resolution=int(checkpoint["resolution"]),
        latent_dim=int(checkpoint["latent_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn(
        args.count,
        model.latent_dim,
        device=device,
        generator=generator,
    ) * args.temperature
    with torch.inference_mode():
        probabilities = torch.sigmoid(model.decode(latent)).cpu().numpy()[:, 0]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, probability_grid in enumerate(probabilities):
        voxels = binarize_voxels(probability_grid, args.threshold)
        stem = output_dir / f"shape_{index:03d}"
        np.save(stem.with_suffix(".npy"), voxels.astype(np.uint8))
        vertices, triangles = voxels_to_obj(voxels, stem.with_suffix(".obj"))
        print(
            f"{stem.with_suffix('.obj')}: {int(voxels.sum())} voxels, "
            f"{vertices} vertices, {triangles} triangles"
        )


if __name__ == "__main__":
    main()
