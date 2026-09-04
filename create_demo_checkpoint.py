from __future__ import annotations

from pathlib import Path

import torch

from tiny3d.image_model import ImageToVoxelNet


def main() -> None:
    torch.manual_seed(20260902)
    model = ImageToVoxelNet(image_size=64, resolution=16, latent_dim=128)
    destination = Path(__file__).resolve().parent / "runs" / "demo_untrained.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "image_size": model.image_size,
            "resolution": model.resolution,
            "latent_dim": model.latent_dim,
            "architecture": model.architecture,
            "epochs": 0,
            "validation_loss": float("nan"),
        },
        destination,
    )
    print(f"Saved untrained demo checkpoint to {destination}")


if __name__ == "__main__":
    main()
