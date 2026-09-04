from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tiny3d.image_data import load_square_image
from tiny3d.image_model import ImageToVoxelNet
from tiny3d.mesh import binarize_voxels, field_to_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a voxel OBJ from one image.")
    parser.add_argument("--checkpoint", default="runs/image_to_3d.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="result.obj")
    parser.add_argument("--threshold", type=float, default=0.45)
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
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ImageToVoxelNet(
        image_size=int(checkpoint["image_size"]),
        resolution=int(checkpoint["resolution"]),
        latent_dim=int(checkpoint["latent_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = load_square_image(args.image, model.image_size)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    voxels = binarize_voxels(probabilities, args.threshold)

    output = Path(args.output)
    if output.suffix.lower() != ".obj":
        output = output.with_suffix(".obj")
    np.save(output.with_suffix(".npy"), voxels.astype(np.uint8))
    vertices, triangles = field_to_obj(probabilities, output, threshold=args.threshold)
    print(
        f"Saved {output}: {int(voxels.sum())} voxels, "
        f"{vertices} vertices, {triangles} triangles"
    )


if __name__ == "__main__":
    main()
