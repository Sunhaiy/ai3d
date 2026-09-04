from __future__ import annotations

import argparse

from tiny3d.data import generate_dataset, save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple synthetic voxel shapes.")
    parser.add_argument("--output", default="data/shapes.npz")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voxels = generate_dataset(args.samples, args.resolution, args.seed)
    save_dataset(args.output, voxels, args.seed)
    occupancy = 100.0 * float(voxels.mean())
    print(f"Saved {len(voxels)} shapes to {args.output}")
    print(f"Shape: {voxels.shape}; mean occupancy: {occupancy:.2f}%")


if __name__ == "__main__":
    main()

