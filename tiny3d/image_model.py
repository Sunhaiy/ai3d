from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


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
        self.architecture = "legacy"
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


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ScalableImageToVoxelNet(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        resolution: int = 256,
        latent_dim: int = 256,
        *,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if resolution < 16 or resolution & (resolution - 1):
            raise ValueError("scalable resolution must be a power of two of at least 16")

        self.image_size = image_size
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.architecture = "scalable"
        self.gradient_checkpointing = gradient_checkpointing
        self.base_size = 4
        self.base_channels = 128
        self.upsample_stages = (resolution // self.base_size).bit_length() - 1
        image_features = 128 * (image_size // 16) ** 2

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
        seed_features = self.base_channels * self.base_size**3
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, seed_features),
            nn.SiLU(),
        )

        decoder_layers: list[nn.Module] = [
            nn.Unflatten(
                1,
                (
                    self.base_channels,
                    self.base_size,
                    self.base_size,
                    self.base_size,
                ),
            )
        ]
        input_channels = self.base_channels
        for stage in range(self.upsample_stages):
            final_stage = stage == self.upsample_stages - 1
            output_channels = 1 if final_stage else max(8, input_channels // 2)
            decoder_layers.append(
                nn.ConvTranspose3d(
                    input_channels,
                    output_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if not final_stage:
                decoder_layers.extend(
                    [
                        nn.GroupNorm(
                            _normalization_groups(output_channels),
                            output_channels,
                        ),
                        nn.SiLU(),
                    ]
                )
            input_channels = output_channels
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        seed = self.from_latent(self.image_encoder(images))
        if self.training and self.gradient_checkpointing:
            return checkpoint_sequential(
                self.decoder,
                self.upsample_stages,
                seed,
                use_reentrant=False,
            )
        return self.decoder(seed)


class ImplicitImageToVoxelNet(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        resolution: int = 512,
        latent_dim: int = 256,
    ):
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if resolution < 512 or resolution & (resolution - 1):
            raise ValueError("implicit resolution must be a power of two of at least 512")

        self.image_size = image_size
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.architecture = "implicit"
        self.frequencies = (1.0, 2.0, 4.0, 8.0, 16.0)
        image_features = 128 * (image_size // 16) ** 2

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
        coordinate_features = 3 + len(self.frequencies) * 6
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim + coordinate_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        features = [points]
        for frequency in self.frequencies:
            scaled = points * (torch.pi * frequency)
            features.extend((torch.sin(scaled), torch.cos(scaled)))
        return torch.cat(features, dim=-1)

    def query_encoded(
        self,
        image_features: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape [batch, count, 3]")
        if points.shape[0] != image_features.shape[0]:
            raise ValueError("image and point batch sizes must match")
        encoded_points = self.encode_coordinates(points)
        expanded_images = image_features.unsqueeze(1).expand(-1, points.shape[1], -1)
        return self.point_decoder(
            torch.cat((expanded_images, encoded_points), dim=-1)
        ).squeeze(-1)

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return self.query_encoded(self.encode_images(images), points)


ImageToVoxelModel = (
    ImageToVoxelNet | ScalableImageToVoxelNet | ImplicitImageToVoxelNet
)


def create_image_to_voxel_model(
    *,
    image_size: int,
    resolution: int,
    latent_dim: int,
    architecture: str | None = None,
    gradient_checkpointing: bool = False,
) -> ImageToVoxelModel:
    selected = architecture or (
        "implicit" if resolution >= 512 else "scalable" if resolution >= 256 else "legacy"
    )
    if selected == "legacy":
        if resolution > 128:
            raise ValueError("legacy image-to-voxel architecture only supports up to 128^3")
        return ImageToVoxelNet(
            image_size=image_size,
            resolution=resolution,
            latent_dim=latent_dim,
        )
    if selected == "scalable":
        if resolution > 256:
            raise ValueError("scalable dense architecture only supports up to 256^3")
        return ScalableImageToVoxelNet(
            image_size=image_size,
            resolution=resolution,
            latent_dim=latent_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
    if selected == "implicit":
        return ImplicitImageToVoxelNet(
            image_size=image_size,
            resolution=resolution,
            latent_dim=latent_dim,
        )
    raise ValueError(f"unknown image-to-voxel architecture: {selected}")
