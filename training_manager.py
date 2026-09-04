from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny3d.image_data import find_images
from tiny3d.mesh_data import find_meshes


ACTIVE_STATUSES = {"preparing", "training", "pausing", "stopping"}
PREPARE_PROGRESS = re.compile(r"\[(?P<current>\d+)/(?P<total>\d+)\]")
BATCH_PROGRESS = re.compile(
    r"epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"batch\s+(?P<batch>\d+)/(?P<batches>\d+):\s+"
    r"train\s+loss=(?P<train_loss>[\d.eE+-]+)"
)
EPOCH_PROGRESS = re.compile(
    r"epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+):\s+"
    r"train\s+loss=(?P<train_loss>[\d.eE+-]+).*?"
    r"(?:validation\s+loss=(?P<validation_loss>[\d.eE+-]+).*?"
    r"iou=(?P<iou>[\d.eE+-]+))?$"
)
DATASET_COUNTS = re.compile(
    r"Device:\s+(?P<device>[^;]+);\s+train:\s+(?P<train>\d+);\s+"
    r"validation:\s+(?P<validation>\d+)"
)
INVALID_RUN_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_run_name(value: str | None, timestamp: datetime | None = None) -> str:
    supplied = bool(value and value.strip())
    name = value.strip() if value else ""
    if name.lower().endswith(".pt"):
        name = name[:-3].rstrip()
    name = name.rstrip(". ")
    if not name:
        if supplied:
            raise ValueError("训练名称不能为空文件名")
        current = timestamp or datetime.now()
        return current.strftime("image3d_%Y%m%d_%H%M%S")
    if len(name) > 80:
        raise ValueError("训练名称不能超过 80 个字符")
    if INVALID_RUN_NAME.search(name) or name in {".", ".."}:
        raise ValueError("训练名称包含 Windows 文件名不允许的字符")
    if name.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("训练名称是 Windows 保留名称")
    return name


def inspect_data_root(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    meshes_dir = root / "meshes"
    images_dir = root / "images"
    if not meshes_dir.is_dir():
        raise ValueError(f"找不到模型目录：{meshes_dir}")
    if not images_dir.is_dir():
        raise ValueError(f"找不到图片目录：{images_dir}")

    meshes = find_meshes(meshes_dir)
    image_index = find_images(images_dir)
    if not meshes:
        raise ValueError("模型目录中没有支持的 3D 文件")

    view_counts = [len(image_index.get(path.stem.casefold(), [])) for path in meshes]
    matched_counts = [count for count in view_counts if count]
    missing = [path.name for path, count in zip(meshes, view_counts) if not count]
    all_mesh_files = sum(1 for path in meshes_dir.rglob("*") if path.is_file())
    return {
        "data_root": str(root),
        "meshes_dir": str(meshes_dir),
        "images_dir": str(images_dir),
        "mesh_count": len(meshes),
        "matched_mesh_count": len(matched_counts),
        "missing_mesh_count": len(missing),
        "missing_meshes": missing[:20],
        "ignored_mesh_count": max(0, all_mesh_files - len(meshes)),
        "image_count": sum(len(paths) for paths in image_index.values()),
        "pair_count": sum(matched_counts),
        "min_views": min(matched_counts, default=0),
        "max_views": max(matched_counts, default=0),
    }


def parse_training_progress(line: str) -> dict[str, Any] | None:
    batch_match = BATCH_PROGRESS.search(line)
    if batch_match:
        values = batch_match.groupdict()
        return {
            "kind": "batch",
            "epoch": int(values["epoch"]),
            "epochs": int(values["epochs"]),
            "batch": int(values["batch"]),
            "batches": int(values["batches"]),
            "train_loss": float(values["train_loss"]),
        }

    epoch_match = EPOCH_PROGRESS.search(line)
    if not epoch_match:
        return None
    values = epoch_match.groupdict()
    return {
        "kind": "epoch",
        "epoch": int(values["epoch"]),
        "epochs": int(values["epochs"]),
        "train_loss": float(values["train_loss"]),
        "validation_loss": (
            float(values["validation_loss"]) if values["validation_loss"] else None
        ),
        "iou": float(values["iou"]) if values["iou"] else None,
    }


class TrainingManager:
    def __init__(self, project_root: Path, default_data_root: Path):
        self.project_root = project_root.resolve()
        self.dataset_path = self.project_root / "data" / "image3d_pairs.npz"
        self.dataset_temp_path = self.project_root / "data" / "image3d_pairs.building.npz"
        self.preview_dir = self.project_root / "data" / "image3d_previews"
        self.checkpoint_path = self.project_root / "runs" / "image_to_3d.pt"
        self.resume_checkpoint_path = self.project_root / "runs" / "image_to_3d.resume"
        self.resume_metadata_path = self.project_root / "runs" / "image_to_3d.resume.json"
        self.pause_request_path = self.project_root / "runs" / "image_to_3d.pause"
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._process_kind: str | None = None
        self._started_monotonic: float | None = None
        self._dataset_inspected = False
        self._gpu_cache: dict[str, Any] | None = None
        self._gpu_checked_at = 0.0
        self._elapsed_before_launch = 0.0
        resume_state = self._resume_file_state()
        resume_config = (
            {
                "epochs": int(resume_state["target_epochs"]),
                "batch_size": int(resume_state["batch_size"]),
                "resolution": int(resume_state["resolution"]),
                "image_size": int(resume_state["image_size"]),
                "run_name": str(resume_state.get("run_name", "")),
                "initial_checkpoint": resume_state.get("initial_checkpoint"),
                "max_hours": float(resume_state.get("max_hours", 0.0)),
                "architecture": str(resume_state.get("architecture", "legacy")),
            }
            if resume_state
            else {
                "epochs": 100,
                "batch_size": 32,
                "resolution": 32,
                "image_size": 128,
                "run_name": "",
                "initial_checkpoint": None,
                "max_hours": 0.0,
                "architecture": "legacy",
            }
        )
        self._state: dict[str, Any] = {
            "status": "paused" if resume_state else "idle",
            "stage": "paused" if resume_state else "idle",
            "progress": float(resume_state.get("progress", 0.0)) if resume_state else 0.0,
            "data_root": str(default_data_root.resolve()),
            "dataset": {
                **self._dataset_file_state(),
                "mesh_count": 0,
                "matched_mesh_count": 0,
                "missing_mesh_count": 0,
                "ignored_mesh_count": 0,
                "image_count": 0,
                "pair_count": 0,
                "min_views": 0,
                "max_views": 0,
            },
            "config": resume_config,
            "current_epoch": int(resume_state.get("epoch", 0)) if resume_state else 0,
            "total_epochs": int(resume_config["epochs"]),
            "current_batch": int(resume_state.get("completed_batches", 0)) if resume_state else 0,
            "total_batches": int(resume_state.get("total_batches", 0)) if resume_state else 0,
            "metrics": resume_state.get("metrics", {}) if resume_state else {
                "train_loss": None,
                "validation_loss": None,
                "iou": None,
            },
            "history": resume_state.get("history", []) if resume_state else [],
            "logs": ([{
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": "发现已保存的暂停训练，可以继续训练",
            }] if resume_state else []),
            "pid": None,
            "started_at": None,
            "updated_at": now_iso(),
            "elapsed_seconds": (
                float(resume_state.get("elapsed_training_seconds", 0.0))
                if resume_state
                else 0.0
            ),
            "error": None,
            "stop_requested": False,
            "pause_requested": False,
            "can_resume": bool(resume_state),
            "output_checkpoint": resume_state.get("output_checkpoint") if resume_state else None,
        }

    def _dataset_file_state(self) -> dict[str, Any]:
        exists = self.dataset_path.is_file()
        result = {
            "ready": exists,
            "path": str(self.dataset_path),
            "size_mb": (
                round(self.dataset_path.stat().st_size / (1024 * 1024), 1) if exists else 0
            ),
            "resolution": 0,
            "image_size": 0,
            "target_count": 0,
            "preview_path": str(self.preview_dir),
        }
        if exists:
            with np.load(self.dataset_path) as archive:
                result["resolution"] = int(
                    archive["resolution"] if "resolution" in archive.files else archive["voxels"].shape[-1]
                )
                result["image_size"] = int(
                    archive["image_size"] if "image_size" in archive.files else archive["images"].shape[1]
                )
                result["target_count"] = int(
                    archive["target_count"] if "target_count" in archive.files else len(archive["voxels"])
                )
        return result

    def _resume_file_state(self) -> dict[str, Any] | None:
        if not self.resume_checkpoint_path.is_file() or not self.resume_metadata_path.is_file():
            return None
        try:
            metadata = json.loads(self.resume_metadata_path.read_text(encoding="utf-8"))
            required = {
                "target_epochs",
                "batch_size",
                "resolution",
                "image_size",
                "epoch",
                "completed_batches",
                "total_batches",
                "output_checkpoint",
            }
            if not required.issubset(metadata):
                return None
            return metadata
        except (OSError, ValueError, TypeError):
            return None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state["status"] in ACTIVE_STATUSES

    @property
    def is_gpu_busy(self) -> bool:
        with self._lock:
            return self._process_kind == "train" and self._state["status"] in ACTIVE_STATUSES

    def _inspect_once(self) -> None:
        with self._lock:
            if self._dataset_inspected:
                return
            data_root = self._state["data_root"]
            self._dataset_inspected = True
        try:
            summary = inspect_data_root(data_root)
            error = None
        except Exception as caught:
            summary = None
            error = str(caught)
        with self._lock:
            if summary:
                self._state["dataset"].update(summary)
            elif error:
                self._state["dataset"]["inspection_error"] = error

    def status(self) -> dict[str, Any]:
        self._inspect_once()
        gpu = self._gpu_status()
        with self._lock:
            result = copy.deepcopy(self._state)
            if self._started_monotonic is not None and result["status"] in ACTIVE_STATUSES:
                result["elapsed_seconds"] = round(
                    self._elapsed_before_launch
                    + time.monotonic()
                    - self._started_monotonic,
                    1,
                )
            result["gpu"] = gpu
            return result

    def start_preparation(
        self,
        data_root: str | Path,
        resolution: int,
        image_size: int,
    ) -> dict[str, Any]:
        if resolution not in {16, 32, 64, 128, 256}:
            raise ValueError("体素分辨率仅支持 16、32、64、128 或 256")
        if image_size not in {64, 128, 256}:
            raise ValueError("图片尺寸仅支持 64、128 或 256")
        summary = inspect_data_root(data_root)
        if summary["missing_mesh_count"]:
            raise ValueError(
                f"有 {summary['missing_mesh_count']} 个模型找不到同名图片，请先修正配对"
            )
        if summary["pair_count"] < 2:
            raise ValueError("至少需要两对图片和 3D 模型")

        with self._lock:
            self._require_idle()
            if self._state["can_resume"]:
                raise RuntimeError("存在已暂停的训练，请先继续训练或结束本次训练")
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_temp_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            str(self.project_root / "prepare_image3d.py"),
            "--meshes",
            summary["meshes_dir"],
            "--images",
            summary["images_dir"],
            "--resolution",
            str(resolution),
            "--image-size",
            str(image_size),
            "--output",
            str(self.dataset_temp_path),
            "--preview-dir",
            str(self.preview_dir),
        ]
        self._launch(
            "prepare",
            command,
            status="preparing",
            stage="preparing",
            data_root=summary["data_root"],
            dataset={
                **self._dataset_file_state(),
                **summary,
                "ready": False,
                "resolution": resolution,
                "image_size": image_size,
                "target_count": summary["matched_mesh_count"],
            },
            config={
                "epochs": 1000 if resolution == 256 else 100,
                "batch_size": 1 if resolution == 256 else 32,
                "resolution": resolution,
                "image_size": image_size,
                "run_name": "",
                "initial_checkpoint": None,
                "max_hours": 72.0 if resolution == 256 else 0.0,
                "architecture": "scalable" if resolution == 256 else "legacy",
            },
            current_epoch=0,
            total_epochs=1000 if resolution == 256 else 100,
            current_batch=0,
            total_batches=0,
            metrics={"train_loss": None, "validation_loss": None, "iou": None},
            history=[],
            output_checkpoint=None,
        )
        return self.status()

    def start_training(self, epochs: int, batch_size: int) -> dict[str, Any]:
        return self.start_named_training(epochs, batch_size, None, None, 0.0)

    def start_named_training(
        self,
        epochs: int,
        batch_size: int,
        run_name: str | None,
        initial_checkpoint: str | None,
        max_hours: float = 0.0,
    ) -> dict[str, Any]:
        if not 1 <= epochs <= 1000:
            raise ValueError("训练轮数必须在 1 到 1000 之间")
        if not 1 <= batch_size <= 128:
            raise ValueError("批大小必须在 1 到 128 之间")
        if not 0.0 <= max_hours <= 720.0:
            raise ValueError("最长训练时长必须在 0 到 720 小时之间")
        if not self.dataset_path.is_file():
            raise ValueError("训练数据集尚未制作完成")
        with self._lock:
            self._require_idle()
            if self._state["can_resume"]:
                raise RuntimeError("存在已暂停的训练，请先继续训练或结束本次训练")
            resolution = int(self._state["dataset"].get("resolution", 16))
            image_size = int(self._state["dataset"].get("image_size", 64))
        initial_path = (
            self._resolve_existing_checkpoint(initial_checkpoint)
            if initial_checkpoint
            else None
        )
        latent_dim = 128 if resolution <= 16 else 256
        architecture = "scalable" if resolution >= 256 else "legacy"
        output_path = self._allocate_checkpoint_path(run_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_resume_files()
        command = [
            sys.executable,
            "-u",
            str(self.project_root / "train_image_to_3d.py"),
            "--data",
            str(self.dataset_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--latent-dim",
            str(latent_dim),
            "--device",
            "cuda",
            "--log-every",
            "20",
            "--output",
            str(output_path),
            "--pause-file",
            str(self.pause_request_path),
            "--resume-checkpoint",
            str(self.resume_checkpoint_path),
            "--max-hours",
            str(max_hours),
        ]
        if initial_path is not None:
            command.extend(["--init-checkpoint", str(initial_path)])
        self._launch(
            "train",
            command,
            status="training",
            stage="training",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "resolution": resolution,
                "image_size": image_size,
                "run_name": output_path.stem,
                "initial_checkpoint": initial_path.name if initial_path else None,
                "max_hours": max_hours,
                "architecture": architecture,
            },
            current_epoch=0,
            total_epochs=epochs,
            current_batch=0,
            total_batches=0,
            metrics={"train_loss": None, "validation_loss": None, "iou": None},
            history=[],
            can_resume=False,
            output_checkpoint=output_path.name,
        )
        return self.status()

    def resume_training(self) -> dict[str, Any]:
        if not self.dataset_path.is_file():
            raise ValueError("训练数据集不存在，无法继续训练")
        resume_state = self._resume_file_state()
        if resume_state is None:
            raise RuntimeError("没有可以继续的暂停训练")
        with self._lock:
            self._require_idle()
            dataset_resolution = int(self._state["dataset"].get("resolution", 0))
            dataset_image_size = int(self._state["dataset"].get("image_size", 0))
            if dataset_resolution != int(resume_state["resolution"]):
                raise ValueError("当前训练集体素分辨率与暂停训练不一致")
            if dataset_image_size != int(resume_state["image_size"]):
                raise ValueError("当前训练集图片尺寸与暂停训练不一致")
            logs = copy.deepcopy(self._state["logs"])
            history = copy.deepcopy(self._state["history"])
            metrics = copy.deepcopy(self._state["metrics"])
            elapsed_seconds = float(self._state["elapsed_seconds"])

        epochs = int(resume_state["target_epochs"])
        batch_size = int(resume_state["batch_size"])
        latent_dim = int(resume_state["latent_dim"])
        max_hours = float(resume_state.get("max_hours", 0.0))
        architecture = str(resume_state.get("architecture", "legacy"))
        output_path = self._checkpoint_output_path(str(resume_state["output_checkpoint"]))
        self.pause_request_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            str(self.project_root / "train_image_to_3d.py"),
            "--data",
            str(self.dataset_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--latent-dim",
            str(latent_dim),
            "--device",
            "cuda",
            "--log-every",
            "20",
            "--output",
            str(output_path),
            "--pause-file",
            str(self.pause_request_path),
            "--resume-checkpoint",
            str(self.resume_checkpoint_path),
            "--resume",
            str(self.resume_checkpoint_path),
            "--max-hours",
            str(max_hours),
        ]
        self._launch(
            "train",
            command,
            status="training",
            stage="training",
            progress=float(resume_state.get("progress", 0.0)),
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "resolution": int(resume_state["resolution"]),
                "image_size": int(resume_state["image_size"]),
                "run_name": str(resume_state.get("run_name", output_path.stem)),
                "initial_checkpoint": resume_state.get("initial_checkpoint"),
                "max_hours": max_hours,
                "architecture": architecture,
            },
            current_epoch=int(resume_state["epoch"]),
            total_epochs=epochs,
            current_batch=int(resume_state["completed_batches"]),
            total_batches=int(resume_state["total_batches"]),
            metrics=metrics,
            history=history,
            logs=logs,
            elapsed_seconds=float(
                resume_state.get("elapsed_training_seconds", elapsed_seconds)
            ),
            can_resume=False,
            output_checkpoint=output_path.name,
        )
        with self._lock:
            self._add_log_locked("从暂停检查点继续训练")
        return self.status()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if (
                self._process is None
                or self._process.poll() is not None
                or self._process_kind != "train"
                or self._state["status"] != "training"
            ):
                raise RuntimeError("当前没有可以暂停的训练")
            self.pause_request_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.pause_request_path.with_suffix(".pause.tmp")
            temporary.write_text("pause\n", encoding="ascii")
            temporary.replace(self.pause_request_path)
            self._state["pause_requested"] = True
            self._state["status"] = "pausing"
            self._state["stage"] = "pausing"
            self._state["updated_at"] = now_iso()
            self._add_log_locked("正在完成当前批次并保存暂停检查点")
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "paused" and self._state["can_resume"]:
                self._clear_resume_files()
                self._state.update(
                    {
                        "status": "stopped",
                        "stage": "stopped",
                        "can_resume": False,
                        "pause_requested": False,
                        "updated_at": now_iso(),
                    }
                )
                self._add_log_locked("已结束暂停训练并删除续训检查点")
                return self.status()
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("当前没有正在运行的训练任务")
            self._state["stop_requested"] = True
            self._state["status"] = "stopping"
            self._state["stage"] = "stopping"
            self._state["updated_at"] = now_iso()
            self._add_log_locked("正在停止任务")
            process = self._process
        try:
            process.terminate()
        except OSError:
            pass
        return self.status()

    def _clear_resume_files(self) -> None:
        self.pause_request_path.unlink(missing_ok=True)
        self.resume_checkpoint_path.unlink(missing_ok=True)
        self.resume_metadata_path.unlink(missing_ok=True)

    def _checkpoint_output_path(self, filename: str) -> Path:
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".pt":
            raise ValueError("检查点文件名无效")
        path = (self.project_root / "runs" / filename).resolve()
        if path.parent != (self.project_root / "runs").resolve():
            raise ValueError("检查点路径无效")
        return path

    def _resolve_existing_checkpoint(self, filename: str) -> Path:
        path = self._checkpoint_output_path(filename)
        if not path.is_file():
            raise ValueError(f"找不到初始模型：{filename}")
        return path

    def _allocate_checkpoint_path(self, run_name: str | None) -> Path:
        stem = normalize_run_name(run_name)
        candidate = self._checkpoint_output_path(f"{stem}.pt")
        suffix = 2
        while candidate.exists():
            candidate = self._checkpoint_output_path(f"{stem}_{suffix:02d}.pt")
            suffix += 1
        return candidate

    def _require_idle(self) -> None:
        if self._state["status"] in ACTIVE_STATUSES:
            raise RuntimeError("已有训练任务正在运行")

    def _launch(self, kind: str, command: list[str], **changes: Any) -> None:
        environment = os.environ.copy()
        environment.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        elapsed_before_launch = float(changes.pop("elapsed_seconds", 0.0))
        with self._lock:
            self._require_idle()
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
            )
            started_at = now_iso()
            self._process = process
            self._process_kind = kind
            self._started_monotonic = time.monotonic()
            self._elapsed_before_launch = elapsed_before_launch
            self._state.update(
                {
                    "progress": 0.0,
                    "logs": [],
                    "elapsed_seconds": elapsed_before_launch,
                    **changes,
                    "pid": process.pid,
                    "started_at": started_at,
                    "updated_at": started_at,
                    "error": None,
                    "stop_requested": False,
                    "pause_requested": False,
                }
            )
            self._add_log_locked(
                "开始制作训练数据" if kind == "prepare" else "开始训练模型"
            )
        threading.Thread(
            target=self._monitor_process,
            args=(kind, process),
            daemon=True,
            name=f"training-{kind}",
        ).start()

    def _monitor_process(self, kind: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                self._record_process_line(kind, line)
        return_code = process.wait()

        stopped = False
        pause_requested = False
        with self._lock:
            stopped = bool(self._state["stop_requested"])
            pause_requested = bool(self._state["pause_requested"])
        if kind == "prepare":
            if return_code == 0 and not stopped and self.dataset_temp_path.is_file():
                self.dataset_temp_path.replace(self.dataset_path)
            else:
                self.dataset_temp_path.unlink(missing_ok=True)
        resume_state = self._resume_file_state() if kind == "train" else None
        if kind == "train" and stopped:
            self._clear_resume_files()
            resume_state = None
        elif kind == "train" and return_code == 0 and not resume_state:
            self._clear_resume_files()

        with self._lock:
            if self._process is not process:
                return
            elapsed = (
                round(
                    self._elapsed_before_launch
                    + time.monotonic()
                    - self._started_monotonic,
                    1,
                )
                if self._started_monotonic is not None
                else self._elapsed_before_launch
            )
            if stopped:
                status = "stopped"
                stage = "stopped"
                self._add_log_locked("任务已停止")
            elif pause_requested and return_code == 0 and resume_state:
                status = "paused"
                stage = "paused"
                self._state.update(
                    {
                        "progress": float(resume_state.get("progress", self._state["progress"])),
                        "current_epoch": int(resume_state["epoch"]),
                        "total_epochs": int(resume_state["target_epochs"]),
                        "current_batch": int(resume_state["completed_batches"]),
                        "total_batches": int(resume_state["total_batches"]),
                        "metrics": resume_state.get("metrics", self._state["metrics"]),
                        "history": resume_state.get("history", self._state["history"]),
                        "can_resume": True,
                    }
                )
                self._add_log_locked("训练已暂停，续训检查点保存完成")
            elif return_code != 0:
                status = "failed"
                stage = "failed"
                self._state["can_resume"] = bool(resume_state)
                self._state["error"] = f"任务退出，代码 {return_code}"
                self._add_log_locked(self._state["error"])
            else:
                status = "completed"
                stage = "dataset_ready" if kind == "prepare" else "checkpoint_ready"
                self._state["progress"] = 100.0
                if kind == "train":
                    self._state["can_resume"] = False
                self._add_log_locked(
                    "训练数据制作完成" if kind == "prepare" else "训练完成，最佳检查点已保存"
                )
                if kind == "prepare":
                    self._state["dataset"].update(self._dataset_file_state())
            self._state.update(
                {
                    "status": status,
                    "stage": stage,
                    "pid": None,
                    "elapsed_seconds": elapsed,
                    "updated_at": now_iso(),
                    "stop_requested": False,
                    "pause_requested": False,
                }
            )
            self._process = None
            self._process_kind = None
            self._started_monotonic = None
            self._elapsed_before_launch = 0.0

    def _record_process_line(self, kind: str, line: str) -> None:
        with self._lock:
            self._add_log_locked(line)
            if kind == "prepare":
                match = PREPARE_PROGRESS.search(line)
                if match:
                    current = int(match.group("current"))
                    total = int(match.group("total"))
                    self._state["progress"] = round(current / max(total, 1) * 100, 1)
                    self._state["current_batch"] = current
                    self._state["total_batches"] = total
                return

            counts = DATASET_COUNTS.search(line)
            if counts:
                train_count = int(counts.group("train"))
                batch_size = int(self._state["config"]["batch_size"])
                self._state["total_batches"] = math.ceil(train_count / batch_size)
                return

            parsed = parse_training_progress(line)
            if not parsed:
                return
            epoch = parsed["epoch"]
            epochs = parsed["epochs"]
            self._state["current_epoch"] = epoch
            self._state["total_epochs"] = epochs
            self._state["metrics"]["train_loss"] = parsed["train_loss"]
            if parsed["kind"] == "batch":
                batch = parsed["batch"]
                batches = parsed["batches"]
                self._state["current_batch"] = batch
                self._state["total_batches"] = batches
                self._state["progress"] = round(
                    ((epoch - 1) + batch / max(batches, 1)) / epochs * 100,
                    2,
                )
                return

            self._state["current_batch"] = self._state["total_batches"]
            self._state["progress"] = round(epoch / epochs * 100, 2)
            self._state["metrics"].update(
                {
                    "validation_loss": parsed["validation_loss"],
                    "iou": parsed["iou"],
                }
            )
            point = {
                "epoch": epoch,
                "train_loss": parsed["train_loss"],
                "validation_loss": parsed["validation_loss"],
                "iou": parsed["iou"],
            }
            history = self._state["history"]
            if history and history[-1]["epoch"] == epoch:
                history[-1] = point
            else:
                history.append(point)

    def _add_log_locked(self, message: str) -> None:
        self._state["logs"].append(
            {"time": datetime.now().strftime("%H:%M:%S"), "message": message}
        )
        if len(self._state["logs"]) > 800:
            del self._state["logs"][:-800]
        self._state["updated_at"] = now_iso()

    def _gpu_status(self) -> dict[str, Any] | None:
        checked_at = time.monotonic()
        if self._gpu_cache is not None and checked_at - self._gpu_checked_at < 1.5:
            return self._gpu_cache
        try:
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
                creationflags=creation_flags,
            )
            values = [int(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
            self._gpu_cache = {
                "utilization": values[0],
                "memory_used_mb": values[1],
                "memory_total_mb": values[2],
                "temperature_c": values[3],
            }
        except Exception:
            self._gpu_cache = None
        self._gpu_checked_at = checked_at
        return self._gpu_cache
