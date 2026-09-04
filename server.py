from __future__ import annotations

import gc
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tiny3d.image_data import load_square_image
from tiny3d.image_model import ImageToVoxelModel, create_image_to_voxel_model
from tiny3d.mesh import binarize_voxels, field_to_obj
from training_manager import TrainingManager


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
JOBS_DIR = ROOT / "web_jobs"
WEB_DIST = ROOT / "web" / "dist"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

RUNS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tiny Image-to-3D Studio", version="1.0.0")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
inference_lock = threading.Lock()
model_cache: dict[str, tuple[float, ImageToVoxelModel, torch.device]] = {}
training_manager = TrainingManager(
    ROOT,
    Path(os.environ.get("AI3D_DATA_ROOT", r"E:\modeldata")),
)


class PrepareTrainingRequest(BaseModel):
    data_root: str = r"E:\modeldata"
    resolution: int = 32
    image_size: int = 128


class StartTrainingRequest(BaseModel):
    epochs: int = 100
    batch_size: int = 32
    max_hours: float = 0.0
    name: str | None = None
    initial_checkpoint: str | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def update_job(job_id: str, **changes: Any) -> None:
    with jobs_lock:
        jobs[job_id].update(changes)
        jobs[job_id]["updated_at"] = now_iso()


def add_log(job_id: str, message: str) -> None:
    with jobs_lock:
        jobs[job_id]["logs"].append(
            {"time": datetime.now().strftime("%H:%M:%S"), "message": message}
        )
        jobs[job_id]["updated_at"] = now_iso()


def resolve_checkpoint(name: str) -> Path:
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="Invalid checkpoint name")
    checkpoint = (RUNS_DIR / name).resolve()
    if checkpoint.parent != RUNS_DIR.resolve() or not checkpoint.is_file():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return checkpoint


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "is_demo": path.name.startswith("demo_untrained"),
    }
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        result.update(
            {
                "epochs": int(checkpoint.get("epochs", 0)),
                "image_size": int(checkpoint["image_size"]),
                "resolution": int(checkpoint["resolution"]),
                "latent_dim": int(checkpoint["latent_dim"]),
                "architecture": str(checkpoint.get("architecture", "legacy")),
                "target_epochs": int(checkpoint.get("target_epochs", checkpoint.get("epochs", 0))),
                "validation_loss": float(checkpoint.get("validation_loss", 0.0)),
                "initial_checkpoint": checkpoint.get("initial_checkpoint"),
            }
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def load_model(path: Path) -> tuple[ImageToVoxelModel, torch.device]:
    cache_key = str(path)
    modified = path.stat().st_mtime
    cached = model_cache.get(cache_key)
    if cached and cached[0] == modified:
        return cached[1], cached[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_image_to_voxel_model(
        image_size=int(checkpoint["image_size"]),
        resolution=int(checkpoint["resolution"]),
        latent_dim=int(checkpoint["latent_dim"]),
        architecture=str(checkpoint.get("architecture", "legacy")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model_cache[cache_key] = (modified, model, device)
    return model, device


def run_generation(job_id: str, checkpoint_path: Path, image_path: Path, threshold: float) -> None:
    started = time.perf_counter()
    try:
        update_job(job_id, status="running", stage="validate", progress=12)
        add_log(job_id, "输入图片已接收")
        with inference_lock:
            update_job(job_id, stage="load", progress=28)
            add_log(job_id, f"加载检查点 {checkpoint_path.name}")
            model, device = load_model(checkpoint_path)

            update_job(job_id, stage="preprocess", progress=45)
            image = load_square_image(image_path, model.image_size)
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            add_log(job_id, f"图片已归一化为 {model.image_size} x {model.image_size}")

            update_job(job_id, stage="inference", progress=66)
            add_log(job_id, f"在 {device.type.upper()} 上执行体素推理")
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            update_job(job_id, stage="mesh", progress=84)
            voxels = binarize_voxels(probabilities, threshold)
            job_dir = JOBS_DIR / job_id
            obj_path = job_dir / "result.obj"
            npy_path = job_dir / "result.npy"
            np.save(npy_path, voxels.astype(np.uint8))
            vertices, triangles = field_to_obj(
                probabilities,
                obj_path,
                threshold=threshold,
            )
            add_log(
                job_id,
                f"网格完成：{int(voxels.sum())} 体素，{vertices} 顶点，{triangles} 三角面",
            )

        elapsed = time.perf_counter() - started
        update_job(
            job_id,
            status="completed",
            stage="complete",
            progress=100,
            elapsed_seconds=round(elapsed, 2),
            voxel_count=int(voxels.sum()),
            vertex_count=vertices,
            triangle_count=triangles,
            result_obj=f"/artifacts/{job_id}/result.obj",
            result_npy=f"/artifacts/{job_id}/result.npy",
        )
        add_log(job_id, f"任务完成，用时 {elapsed:.2f} 秒")
    except Exception as error:
        with jobs_lock:
            failed_stage = jobs[job_id]["stage"]
        update_job(
            job_id,
            status="failed",
            stage="failed",
            failed_stage=failed_stage,
            error=str(error),
        )
        add_log(job_id, f"任务失败：{error}")


@app.get("/api/system")
def system_status() -> dict[str, Any]:
    checkpoints = [
        checkpoint_metadata(path)
        for path in sorted(RUNS_DIR.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    ]
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "checkpoints": checkpoints,
        "training_active": training_manager.is_active,
    }


@app.get("/api/training")
def training_status() -> dict[str, Any]:
    return training_manager.status()


@app.post("/api/training/prepare", status_code=202)
def prepare_training_data(request: PrepareTrainingRequest) -> dict[str, Any]:
    try:
        return training_manager.start_preparation(
            request.data_root,
            request.resolution,
            request.image_size,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/training/start", status_code=202)
def start_training(request: StartTrainingRequest) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise HTTPException(status_code=409, detail="CUDA 不可用，无法开始 GPU 训练")
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="图片生成任务正在占用 GPU，请稍后再试")
    try:
        model_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return training_manager.start_named_training(
            request.epochs,
            request.batch_size,
            request.name,
            request.initial_checkpoint,
            request.max_hours,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    finally:
        inference_lock.release()


@app.post("/api/training/stop", status_code=202)
def stop_training() -> dict[str, Any]:
    try:
        return training_manager.stop()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/training/pause", status_code=202)
def pause_training() -> dict[str, Any]:
    try:
        return training_manager.pause()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/training/resume", status_code=202)
def resume_training() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise HTTPException(status_code=409, detail="CUDA 不可用，无法继续 GPU 训练")
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="图片生成任务正在占用 GPU，请稍后再试")
    try:
        model_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return training_manager.resume_training()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    finally:
        inference_lock.release()


@app.post("/api/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    checkpoint: str = Form(...),
    threshold: float = Form(0.45),
) -> dict[str, Any]:
    if training_manager.is_gpu_busy:
        raise HTTPException(status_code=409, detail="模型正在训练，暂时不能执行图片生成")
    checkpoint_path = resolve_checkpoint(checkpoint)
    extension = Path(image.filename or "").suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    if not 0.05 <= threshold <= 0.95:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.05 and 0.95")

    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    image_path = job_dir / f"input{extension}"
    image_path.write_bytes(content)
    created_at = now_iso()
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "upload",
        "progress": 5,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "input_url": f"/artifacts/{job_id}/{image_path.name}",
        "created_at": created_at,
        "updated_at": created_at,
        "logs": [],
        "error": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    background_tasks.add_task(run_generation, job_id, checkpoint_path, image_path, threshold)
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job, logs=list(job["logs"]))


app.mount("/artifacts", StaticFiles(directory=JOBS_DIR), name="artifacts")
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
