from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tiny3d.image_data import ImageVoxelDataset
from tiny3d.image_model import ImageToVoxelNet
from tiny3d.model import voxel_reconstruction_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small single-image to voxel model.")
    parser.add_argument("--data", default="data/image3d_pairs.npz")
    parser.add_argument("--output", default="runs/image_to_3d.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print batch progress every N batches; 0 only prints epoch summaries.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--pause-file", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def average_metrics(totals: dict[str, float], batches: int) -> str:
    return " ".join(f"{key}={value / batches:.4f}" for key, value in totals.items())


def split_by_mesh(
    dataset: ImageVoxelDataset,
    validation_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if validation_split == 0.0:
        return Subset(dataset, list(range(len(dataset)))), Subset(dataset, [])

    if dataset.sources is None:
        groups = {str(index): [index] for index in range(len(dataset))}
    else:
        groups: dict[str, list[int]] = {}
        for index, source in enumerate(dataset.sources):
            mesh_source = source.rsplit("|", 1)[-1]
            groups.setdefault(mesh_source, []).append(index)
    if len(groups) < 2:
        return Subset(dataset, list(range(len(dataset)))), Subset(dataset, [])

    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    validation_groups = max(1, int(round(len(group_names) * validation_split)))
    validation_groups = min(validation_groups, len(group_names) - 1)
    validation_names = set(group_names[:validation_groups])
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for name, indices in groups.items():
        destination = validation_indices if name in validation_names else training_indices
        destination.extend(indices)
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def evaluate(
    model: ImageToVoxelNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, str, dict[str, float]]:
    model.eval()
    totals = {key: 0.0 for key in ("loss", "bce", "dice", "iou")}
    with torch.inference_mode():
        for images, voxels in loader:
            images = images.to(device, non_blocking=True)
            voxels = voxels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss, metrics = voxel_reconstruction_loss(logits, voxels)
            predicted = torch.sigmoid(logits) >= 0.5
            target = voxels >= 0.5
            intersection = (predicted & target).sum(dim=(1, 2, 3, 4)).float()
            union = (predicted | target).sum(dim=(1, 2, 3, 4)).float()
            iou = ((intersection + 1.0) / (union + 1.0)).mean()
            for key in ("loss", "bce", "dice"):
                totals[key] += metrics[key]
            totals["iou"] += float(iou)
    averages = {key: value / len(loader) for key, value in totals.items()}
    return averages["loss"], average_metrics(totals, len(loader)), averages


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def save_checkpoint(
    destination: Path,
    model: ImageToVoxelNet,
    epoch: int,
    validation_loss: float,
    *,
    target_epochs: int,
    initial_checkpoint: str | None,
) -> None:
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "image_size": model.image_size,
            "resolution": model.resolution,
            "latent_dim": model.latent_dim,
            "epochs": epoch,
            "target_epochs": target_epochs,
            "validation_loss": validation_loss,
            "initial_checkpoint": initial_checkpoint,
        },
        destination,
    )


def resume_metadata_path(checkpoint: Path) -> Path:
    return Path(f"{checkpoint}.json")


def save_resume_checkpoint(
    destination: Path,
    model: ImageToVoxelNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    completed_batches: int,
    total_batches: int,
    epoch_totals: dict[str, float],
    best_loss: float,
    target_epochs: int,
    batch_size: int,
    history: list[dict[str, float | int | None]],
    output_checkpoint: str,
    initial_checkpoint: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "image_size": model.image_size,
            "resolution": model.resolution,
            "latent_dim": model.latent_dim,
            "epoch": epoch,
            "completed_batches": completed_batches,
            "epoch_totals": epoch_totals,
            "best_loss": best_loss,
            "target_epochs": target_epochs,
            "batch_size": batch_size,
            "history": history,
            "output_checkpoint": output_checkpoint,
            "run_name": Path(output_checkpoint).stem,
            "initial_checkpoint": initial_checkpoint,
        },
        destination,
    )
    train_loss = (
        epoch_totals["loss"] / completed_batches if completed_batches else None
    )
    metadata = {
        "version": 1,
        "image_size": model.image_size,
        "resolution": model.resolution,
        "latent_dim": model.latent_dim,
        "epoch": epoch,
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "target_epochs": target_epochs,
        "batch_size": batch_size,
        "output_checkpoint": output_checkpoint,
        "run_name": Path(output_checkpoint).stem,
        "initial_checkpoint": initial_checkpoint,
        "progress": round(
            ((epoch - 1) + completed_batches / max(total_batches, 1))
            / target_epochs
            * 100,
            2,
        ),
        "metrics": {
            "train_loss": train_loss,
            "validation_loss": history[-1]["validation_loss"] if history else None,
            "iou": history[-1]["iou"] if history else None,
        },
        "history": history,
    }
    metadata_destination = resume_metadata_path(destination)
    temporary = metadata_destination.with_suffix(metadata_destination.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    temporary.replace(metadata_destination)


def load_resume_checkpoint(
    source: Path,
    model: ImageToVoxelNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    target_epochs: int,
    batch_size: int,
    output_checkpoint: str,
) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    expected = {
        "image_size": model.image_size,
        "resolution": model.resolution,
        "latent_dim": model.latent_dim,
        "target_epochs": target_epochs,
        "batch_size": batch_size,
    }
    for key, value in expected.items():
        if int(checkpoint.get(key, -1)) != value:
            raise ValueError(
                f"resume checkpoint {key}={checkpoint.get(key)!r} does not match {value}"
            )
    if checkpoint.get("output_checkpoint") != output_checkpoint:
        raise ValueError("resume checkpoint output file does not match this training run")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def load_initial_checkpoint(source: Path, model: ImageToVoxelNet) -> tuple[int, int]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    if not compatible_state:
        raise ValueError("initial checkpoint has no compatible model parameters")
    model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), len(target_state)


def training_loader_for_epoch(
    training_set: Subset,
    loader_options: dict[str, Any],
    seed: int,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(training_set, shuffle=True, generator=generator, **loader_options)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("validation-split must be in [0, 1)")
    if args.log_every < 0:
        raise ValueError("log-every cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    dataset = ImageVoxelDataset(args.data)
    if len(dataset) < 2:
        raise ValueError("at least two image/3D pairs are required")
    training_set, validation_set = split_by_mesh(dataset, args.validation_split, args.seed)
    training_count = len(training_set)
    validation_count = len(validation_set)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    validation_loader = (
        DataLoader(validation_set, shuffle=False, **loader_options)
        if validation_count
        else None
    )
    image_size = dataset.images.shape[-1]
    resolution = dataset.voxels.shape[-1]
    model = ImageToVoxelNet(
        image_size=image_size,
        resolution=resolution,
        latent_dim=args.latent_dim,
    ).to(device)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    initial_checkpoint_name = (
        Path(args.init_checkpoint).name if args.init_checkpoint else None
    )
    if args.init_checkpoint:
        initial_checkpoint_path = Path(args.init_checkpoint)
        if not initial_checkpoint_path.is_file():
            raise ValueError(
                f"initial checkpoint does not exist: {initial_checkpoint_path}"
            )
        loaded_parameters, total_parameters = load_initial_checkpoint(
            initial_checkpoint_path,
            model,
        )
        print(
            f"Initialized model from {initial_checkpoint_path}; loaded "
            f"{loaded_parameters}/{total_parameters} compatible tensors",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pause_file = Path(args.pause_file) if args.pause_file else None
    resume_destination = (
        Path(args.resume_checkpoint) if args.resume_checkpoint else None
    )
    if pause_file is not None and resume_destination is None:
        raise ValueError("--resume-checkpoint is required when --pause-file is used")

    best_loss = float("inf")
    start_epoch = 1
    resume_batches = 0
    resume_totals = {key: 0.0 for key in ("loss", "bce", "dice")}
    history: list[dict[str, float | int | None]] = []
    if args.resume:
        resume_source = Path(args.resume)
        if not resume_source.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume_source}")
        resume_state = load_resume_checkpoint(
            resume_source,
            model,
            optimizer,
            scaler,
            target_epochs=args.epochs,
            batch_size=args.batch_size,
            output_checkpoint=destination.name,
        )
        start_epoch = int(resume_state["epoch"])
        resume_batches = int(resume_state["completed_batches"])
        resume_totals = {
            key: float(resume_state["epoch_totals"][key])
            for key in ("loss", "bce", "dice")
        }
        best_loss = float(resume_state["best_loss"])
        history = list(resume_state.get("history", []))
        initial_checkpoint_name = resume_state.get("initial_checkpoint")
        print(
            f"Resuming epoch {start_epoch:03d}/{args.epochs} after "
            f"batch {resume_batches:03d}",
            flush=True,
        )

    print(
        f"Device: {device}; train: {training_count}; validation: {validation_count}; "
        f"image: {image_size}^2; voxels: {resolution}^3"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        training_loader = training_loader_for_epoch(
            training_set,
            loader_options,
            args.seed,
            epoch,
        )
        model.train()
        completed_batches = resume_batches if epoch == start_epoch else 0
        totals = (
            dict(resume_totals)
            if epoch == start_epoch
            else {key: 0.0 for key in ("loss", "bce", "dice")}
        )
        for batch_index, (images, voxels) in enumerate(training_loader, start=1):
            if batch_index <= completed_batches:
                continue
            images = images.to(device, non_blocking=True)
            voxels = voxels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss, metrics = voxel_reconstruction_loss(logits, voxels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] += metrics[key]
            completed_batches = batch_index
            if args.log_every and (
                batch_index % args.log_every == 0 or batch_index == len(training_loader)
            ):
                print(
                    f"epoch {epoch:03d}/{args.epochs} batch "
                    f"{batch_index:03d}/{len(training_loader)}: train "
                    f"{average_metrics(totals, batch_index)}",
                    flush=True,
                )

            if pause_file is not None and pause_file.is_file():
                assert resume_destination is not None
                save_resume_checkpoint(
                    resume_destination,
                    model,
                    optimizer,
                    scaler,
                    epoch=epoch,
                    completed_batches=completed_batches,
                    total_batches=len(training_loader),
                    epoch_totals=totals,
                    best_loss=best_loss,
                    target_epochs=args.epochs,
                    batch_size=args.batch_size,
                    history=history,
                    output_checkpoint=destination.name,
                    initial_checkpoint=initial_checkpoint_name,
                )
                pause_file.unlink(missing_ok=True)
                print(
                    f"Paused at epoch {epoch:03d}/{args.epochs} batch "
                    f"{completed_batches:03d}/{len(training_loader)}; "
                    f"resume checkpoint saved to {resume_destination}",
                    flush=True,
                )
                return

        train_summary = average_metrics(totals, len(training_loader))
        if validation_loader is not None:
            validation_loss, validation_summary, validation_metrics = evaluate(
                model,
                validation_loader,
                device,
            )
            print(
                f"epoch {epoch:03d}/{args.epochs}: train {train_summary} | "
                f"validation {validation_summary}"
            )
            validation_iou: float | None = validation_metrics["iou"]
        else:
            validation_loss = totals["loss"] / len(training_loader)
            validation_iou = None
            print(f"epoch {epoch:03d}/{args.epochs}: train {train_summary}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": totals["loss"] / len(training_loader),
                "validation_loss": validation_loss,
                "iou": validation_iou,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                destination,
                model,
                epoch,
                validation_loss,
                target_epochs=args.epochs,
                initial_checkpoint=initial_checkpoint_name,
            )

        if (
            epoch < args.epochs
            and pause_file is not None
            and pause_file.is_file()
        ):
            assert resume_destination is not None
            save_resume_checkpoint(
                resume_destination,
                model,
                optimizer,
                scaler,
                epoch=epoch + 1,
                completed_batches=0,
                total_batches=len(training_loader),
                epoch_totals={key: 0.0 for key in ("loss", "bce", "dice")},
                best_loss=best_loss,
                target_epochs=args.epochs,
                batch_size=args.batch_size,
                history=history,
                output_checkpoint=destination.name,
                initial_checkpoint=initial_checkpoint_name,
            )
            pause_file.unlink(missing_ok=True)
            print(
                f"Paused before epoch {epoch + 1:03d}/{args.epochs}; "
                f"resume checkpoint saved to {resume_destination}",
                flush=True,
            )
            return

        resume_batches = 0
        resume_totals = {key: 0.0 for key in ("loss", "bce", "dice")}

    if resume_destination is not None:
        resume_destination.unlink(missing_ok=True)
        resume_metadata_path(resume_destination).unlink(missing_ok=True)
    if pause_file is not None:
        pause_file.unlink(missing_ok=True)
    print(f"Saved best checkpoint to {destination} (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
