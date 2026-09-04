from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VoxelVAE(nn.Module):
    def __init__(self, resolution: int = 16, latent_dim: int = 8):
        super().__init__()
        if resolution < 8 or resolution % 8 != 0:
            raise ValueError("resolution must be at least 8 and divisible by 8")

        self.resolution = resolution
        self.latent_dim = latent_dim
        self.encoded_size = resolution // 8
        encoded_features = 64 * self.encoded_size**3

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(encoded_features, latent_dim)
        self.to_logvar = nn.Linear(encoded_features, latent_dim)
        self.from_latent = nn.Linear(latent_dim, encoded_features)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, self.encoded_size, self.encoded_size, self.encoded_size)),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, voxels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(voxels)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latent))

    def forward(
        self, voxels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(voxels)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def voxel_reconstruction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_weight = torch.tensor(2.0, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)

    probabilities = torch.sigmoid(logits)
    reduce_dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    total = bce + dice
    metrics = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
    }
    return total, metrics


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction, metrics = voxel_reconstruction_loss(logits, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    total = reconstruction + beta * kl
    metrics["loss"] = float(total.detach())
    metrics["kl"] = float(kl.detach())
    return total, metrics
