import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from tiny3d.data import generate_dataset, save_dataset, VoxelDataset
from tiny3d.image_data import ImageVoxelDataset, render_voxel_projection, save_image_voxel_dataset
from tiny3d.image_model import (
    ImageToVoxelNet,
    ScalableImageToVoxelNet,
    create_image_to_voxel_model,
)
from tiny3d.mesh import binarize_voxels, field_to_obj, voxels_to_obj
from tiny3d.mesh_data import find_meshes, load_mesh, mesh_to_voxels
from tiny3d.model import VoxelVAE, vae_loss, voxel_reconstruction_loss
from train_image_to_3d import load_initial_checkpoint, split_by_mesh
from training_manager import (
    TrainingManager,
    inspect_data_root,
    normalize_run_name,
    parse_training_progress,
)


def test_dataset_round_trip(tmp_path):
    voxels = generate_dataset(samples=4, resolution=16, seed=7)
    assert voxels.shape == (4, 16, 16, 16)
    assert voxels.dtype == np.uint8
    assert np.all(voxels.sum(axis=(1, 2, 3)) > 0)

    path = tmp_path / "shapes.npz"
    save_dataset(path, voxels, seed=7)
    dataset = VoxelDataset(path)
    assert dataset[0].shape == (1, 16, 16, 16)
    assert dataset[0].dtype == torch.float32


def test_model_forward_and_loss():
    model = VoxelVAE(resolution=16, latent_dim=8)
    target = torch.randint(0, 2, (2, 1, 16, 16, 16), dtype=torch.float32)
    logits, mu, logvar = model(target)
    loss, metrics = vae_loss(logits, target, mu, logvar, beta=0.01)
    assert logits.shape == target.shape
    assert mu.shape == (2, 8)
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "bce", "dice", "kl"}


def test_binarize_and_obj_export(tmp_path):
    probabilities = np.zeros((8, 8, 8), dtype=np.float32)
    probabilities[2:5, 2:5, 2:5] = 0.9
    voxels = binarize_voxels(probabilities, threshold=0.5)
    output = tmp_path / "shape.obj"
    vertices, triangles = voxels_to_obj(voxels, output)
    text = output.read_text(encoding="ascii")
    assert vertices > 0
    assert triangles > 0
    assert "\nv " in text
    assert "\nf " in text


def test_probability_field_exports_smooth_surface(tmp_path):
    coordinates = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    probabilities = np.clip(1.3 - np.sqrt(x**2 + y**2 + z**2), 0.0, 1.0)
    output = tmp_path / "smooth.obj"
    vertices, triangles = field_to_obj(probabilities, output, threshold=0.5)
    text = output.read_text(encoding="ascii")
    assert vertices > 100
    assert triangles > 100
    assert "marching cubes" in text


def test_mesh_dataset_conversion(tmp_path):
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    mesh_path = tmp_path / "nested" / "box.obj"
    mesh_path.parent.mkdir()
    mesh.export(mesh_path)
    three_mf_path = tmp_path / "box.3mf"
    mesh.export(three_mf_path)

    assert find_meshes(tmp_path) == [three_mf_path, mesh_path]
    assert isinstance(load_mesh(three_mf_path), trimesh.Trimesh)
    voxels = mesh_to_voxels(mesh, resolution=16)
    assert voxels.shape == (16, 16, 16)
    assert voxels.dtype == np.uint8
    assert 0 < voxels.sum() < voxels.size


def test_image_to_voxel_pipeline(tmp_path):
    voxels = generate_dataset(samples=3, resolution=16, seed=11)
    images = np.stack([render_voxel_projection(voxel, image_size=64) for voxel in voxels])
    data_path = tmp_path / "pairs.npz"
    save_image_voxel_dataset(
        data_path,
        images,
        voxels,
        sources=["one", "two", "three"],
    )
    dataset = ImageVoxelDataset(data_path)
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target.shape == (1, 16, 16, 16)
    assert dataset.sources == ["one", "two", "three"]

    model = ImageToVoxelNet(image_size=64, resolution=16, latent_dim=16)
    logits = model(torch.stack([dataset[0][0], dataset[1][0]]))
    targets = torch.stack([dataset[0][1], dataset[1][1]])
    loss, metrics = voxel_reconstruction_loss(logits, targets)
    assert logits.shape == targets.shape
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "bce", "dice"}


def test_scalable_image_to_voxel_model_forward_and_backward():
    model = ScalableImageToVoxelNet(
        image_size=32,
        resolution=32,
        latent_dim=16,
        gradient_checkpointing=True,
    )
    images = torch.randn(1, 3, 32, 32)
    logits = model(images)
    logits.mean().backward()
    assert logits.shape == (1, 1, 32, 32, 32)
    assert model.image_encoder[0].weight.grad is not None


def test_model_factory_keeps_256_parameter_count_bounded():
    legacy = create_image_to_voxel_model(
        image_size=64,
        resolution=32,
        latent_dim=256,
    )
    scalable = create_image_to_voxel_model(
        image_size=128,
        resolution=256,
        latent_dim=256,
    )
    assert legacy.architecture == "legacy"
    assert scalable.architecture == "scalable"
    assert sum(parameter.numel() for parameter in scalable.parameters()) < 10_000_000


def test_legacy_checkpoint_metadata_defaults_to_legacy_architecture():
    checkpoint = {"image_size": 64, "resolution": 16, "latent_dim": 16}
    model = create_image_to_voxel_model(
        image_size=checkpoint["image_size"],
        resolution=checkpoint["resolution"],
        latent_dim=checkpoint["latent_dim"],
        architecture=str(checkpoint.get("architecture", "legacy")),
    )
    assert isinstance(model, ImageToVoxelNet)


def test_initial_checkpoint_can_transfer_compatible_layers(tmp_path):
    source_model = ImageToVoxelNet(image_size=32, resolution=8, latent_dim=4)
    with torch.no_grad():
        source_model.image_encoder[0].weight.fill_(0.25)
    source_path = tmp_path / "source.pt"
    torch.save(
        {
            "model_state": source_model.state_dict(),
            "image_size": 32,
            "resolution": 8,
            "latent_dim": 4,
        },
        source_path,
    )
    target_model = ImageToVoxelNet(image_size=64, resolution=16, latent_dim=8)
    loaded, total = load_initial_checkpoint(source_path, target_model)
    assert 0 < loaded < total
    assert torch.all(target_model.image_encoder[0].weight == 0.25)


def test_initial_checkpoint_transfers_encoder_between_architectures(tmp_path):
    source_model = ImageToVoxelNet(image_size=32, resolution=16, latent_dim=16)
    with torch.no_grad():
        source_model.image_encoder[0].weight.fill_(0.125)
    source_path = tmp_path / "legacy.pt"
    torch.save({"model_state": source_model.state_dict()}, source_path)
    target_model = ScalableImageToVoxelNet(
        image_size=32,
        resolution=32,
        latent_dim=16,
    )
    loaded, total = load_initial_checkpoint(source_path, target_model)
    assert 0 < loaded < total
    assert torch.all(target_model.image_encoder[0].weight == 0.125)


def test_compact_dataset_reuses_voxel_targets(tmp_path):
    voxels = generate_dataset(samples=2, resolution=16, seed=17)
    rendered = [render_voxel_projection(voxel, image_size=64) for voxel in voxels]
    images = np.stack([rendered[0], rendered[0], rendered[1], rendered[1]])
    data_path = tmp_path / "compact_pairs.npz"
    save_image_voxel_dataset(
        data_path,
        images,
        voxels,
        sources=["a|mesh-a", "b|mesh-a", "a|mesh-b", "b|mesh-b"],
        voxel_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    dataset = ImageVoxelDataset(data_path)
    assert len(dataset) == 4
    assert len(dataset.voxels) == 2
    assert torch.equal(dataset[0][1], dataset[1][1])
    assert torch.equal(dataset[2][1], dataset[3][1])


def test_validation_split_keeps_mesh_views_together(tmp_path):
    voxels = generate_dataset(samples=4, resolution=16, seed=13)
    images = np.stack([render_voxel_projection(voxel, image_size=64) for voxel in voxels])
    data_path = tmp_path / "grouped_pairs.npz"
    sources = ["front|mesh-a", "side|mesh-a", "front|mesh-b", "side|mesh-b"]
    save_image_voxel_dataset(data_path, images, voxels, sources=sources)
    dataset = ImageVoxelDataset(data_path)
    training, validation = split_by_mesh(dataset, validation_split=0.5, seed=1)
    training_meshes = {sources[index].rsplit("|", 1)[-1] for index in training.indices}
    validation_meshes = {sources[index].rsplit("|", 1)[-1] for index in validation.indices}
    assert training_meshes.isdisjoint(validation_meshes)


def test_training_data_inspection_and_progress_parsing(tmp_path):
    meshes = tmp_path / "meshes"
    images = tmp_path / "images"
    meshes.mkdir()
    images.mkdir()
    trimesh.creation.box().export(meshes / "figure.3mf")
    (images / "figure__front.png").write_bytes(b"image-placeholder")
    (images / "figure__back.png").write_bytes(b"image-placeholder")

    summary = inspect_data_root(tmp_path)
    assert summary["mesh_count"] == 1
    assert summary["matched_mesh_count"] == 1
    assert summary["pair_count"] == 2

    batch = parse_training_progress(
        "epoch 003/100 batch 020/338: train loss=0.4321 bce=0.1 dice=0.2"
    )
    assert batch == {
        "kind": "batch",
        "epoch": 3,
        "epochs": 100,
        "batch": 20,
        "batches": 338,
        "train_loss": 0.4321,
    }
    epoch = parse_training_progress(
        "epoch 003/100: train loss=0.4000 bce=0.1 dice=0.2 | "
        "validation loss=0.4500 bce=0.1 dice=0.2 iou=0.3200"
    )
    assert epoch is not None
    assert epoch["validation_loss"] == 0.45
    assert epoch["iou"] == 0.32


def test_training_can_pause_and_resume_from_a_batch(tmp_path):
    voxels = generate_dataset(samples=4, resolution=8, seed=23)
    images = np.stack([render_voxel_projection(voxel, image_size=32) for voxel in voxels])
    data_path = tmp_path / "pairs.npz"
    save_image_voxel_dataset(
        data_path,
        images,
        voxels,
        sources=[f"view|mesh-{index}" for index in range(4)],
    )
    output_path = tmp_path / "best.pt"
    resume_path = tmp_path / "training.resume"
    pause_path = tmp_path / "training.pause"
    pause_path.write_text("pause\n", encoding="ascii")
    project_root = Path(__file__).resolve().parents[1]
    base_command = [
        sys.executable,
        str(project_root / "train_image_to_3d.py"),
        "--data",
        str(data_path),
        "--output",
        str(output_path),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--latent-dim",
        "4",
        "--validation-split",
        "0",
        "--device",
        "cpu",
        "--pause-file",
        str(pause_path),
        "--resume-checkpoint",
        str(resume_path),
    ]

    paused = subprocess.run(
        base_command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert paused.returncode == 0, paused.stdout + paused.stderr
    assert "Paused at epoch" in paused.stdout
    assert resume_path.is_file()
    metadata = json.loads(Path(f"{resume_path}.json").read_text(encoding="utf-8"))
    assert metadata["epoch"] == 1
    assert metadata["completed_batches"] == 1
    assert metadata["progress"] == 50.0

    resumed = subprocess.run(
        [*base_command, "--resume", str(resume_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resuming epoch 001/1 after batch 001" in resumed.stdout
    assert output_path.is_file()
    assert not resume_path.exists()
    assert not Path(f"{resume_path}.json").exists()

    continued_path = tmp_path / "continued.pt"
    continued = subprocess.run(
        [
            sys.executable,
            str(project_root / "train_image_to_3d.py"),
            "--data",
            str(data_path),
            "--output",
            str(continued_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--latent-dim",
            "4",
            "--validation-split",
            "0",
            "--device",
            "cpu",
            "--init-checkpoint",
            str(output_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert continued.returncode == 0, continued.stdout + continued.stderr
    continued_checkpoint = torch.load(continued_path, map_location="cpu", weights_only=True)
    assert continued_checkpoint["initial_checkpoint"] == output_path.name


def test_training_manager_restores_paused_state_after_restart(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "image_to_3d.resume").write_bytes(b"checkpoint-placeholder")
    (runs / "image_to_3d.resume.json").write_text(
        json.dumps(
            {
                "target_epochs": 20,
                "batch_size": 4,
                "resolution": 128,
                "image_size": 128,
                "latent_dim": 256,
                "epoch": 3,
                "completed_batches": 7,
                "total_batches": 50,
                "progress": 10.7,
                "output_checkpoint": "named_run.pt",
                "run_name": "named_run",
                "initial_checkpoint": None,
                "metrics": {
                    "train_loss": 0.4,
                    "validation_loss": 0.5,
                    "iou": 0.2,
                },
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    manager = TrainingManager(tmp_path, tmp_path / "modeldata")
    status = manager.status()
    assert status["status"] == "paused"
    assert status["can_resume"] is True
    assert status["current_epoch"] == 3
    assert status["current_batch"] == 7
    assert status["config"]["resolution"] == 128
    assert status["config"]["architecture"] == "legacy"
    assert status["config"]["max_hours"] == 0.0
    assert status["output_checkpoint"] == "named_run.pt"


def test_training_manager_configures_256_long_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "image3d_pairs.npz",
        resolution=np.asarray(256),
        image_size=np.asarray(128),
        target_count=np.asarray(2),
    )
    manager = TrainingManager(tmp_path, tmp_path / "modeldata")
    captured: dict[str, object] = {}

    def capture_launch(kind, command, **changes):
        captured.update({"kind": kind, "command": command, "changes": changes})

    monkeypatch.setattr(manager, "_launch", capture_launch)
    manager.start_named_training(1000, 1, "long_run", None, 72.0)

    command = captured["command"]
    changes = captured["changes"]
    assert command[command.index("--max-hours") + 1] == "72.0"
    assert changes["config"]["architecture"] == "scalable"
    assert changes["config"]["max_hours"] == 72.0


def test_training_run_names_are_safe_and_do_not_overwrite(tmp_path):
    fixed_time = datetime(2026, 9, 4, 16, 5, 7)
    assert normalize_run_name(None, fixed_time) == "image3d_20260904_160507"
    assert normalize_run_name("chair_run.pt") == "chair_run"
    with pytest.raises(ValueError):
        normalize_run_name("../outside")

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "chair_run.pt").write_bytes(b"existing")
    manager = TrainingManager(tmp_path, tmp_path / "modeldata")
    allocated = manager._allocate_checkpoint_path("chair_run")
    assert allocated.name == "chair_run_02.pt"
