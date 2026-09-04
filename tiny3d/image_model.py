from __future__ import annotations

import torch
from torch import nn


class ImageToVoxelNet(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        resolution: int = 16,
        latent_dim: int = 128,
    ):
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if resolution < 8 or resolution % 8 != 0:
            raise ValueError("resolution must be at least 8 and divisible by 8")

        self.image_size = image_size
        self.resolution = resolution
        self.latent_dim = latent_dim
        image_features = 128 * (image_size // 16) ** 2
        encoded_size = resolution // 8
        voxel_features = 64 * encoded_size**3

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(image_features, latent_dim),
            nn.SiLU(),
        )
        self.to_voxels = nn.Sequential(
            nn.Linear(latent_dim, voxel_features),
            nn.SiLU(),
            nn.Unflatten(1, (64, encoded_size, encoded_size, encoded_size)),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.to_voxels(self.image_encoder(images))

