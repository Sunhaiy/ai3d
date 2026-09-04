from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tiny3d.data import VoxelDataset
from tiny3d.model import VoxelVAE, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny 3D voxel VAE.")
    parser.add_argument("--data", default="data/shapes.npz")
    parser.add_argument("--output", default="runs/tiny3d.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.save_every < 1:
        raise ValueError("save-every must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    dataset = VoxelDataset(args.data)
    resolution = dataset.voxels.shape[-1]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = VoxelVAE(resolution=resolution, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; samples: {len(dataset)}; resolution: {resolution}^3")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "bce", "dice", "kl")}
        for voxels in loader:
            voxels = voxels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, mu, logvar = model(voxels)
                beta = args.beta * min(1.0, epoch / max(1, args.epochs * 0.2))
                loss, metrics = vae_loss(logits, voxels, mu, logvar, beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in metrics.items():
                totals[key] += value

        summary = " ".join(f"{key}={value / len(loader):.4f}" for key, value in totals.items())
        print(f"epoch {epoch:03d}/{args.epochs}: {summary}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "resolution": resolution,
                    "latent_dim": args.latent_dim,
                    "seed": args.seed,
                    "epochs": epoch,
                },
                temporary,
            )
            temporary.replace(destination)
            print(f"Saved checkpoint at epoch {epoch} to {destination}")


if __name__ == "__main__":
    main()
