import {
  Button,
  Chip,
  Label,
  ListBox,
  ProgressBar,
  Select,
  Slider,
} from "@heroui/react";
import {
  Box,
  BrainCircuit,
  Check,
  CircleDashed,
  Cpu,
  Download,
  FileBox,
  Image as ImageIcon,
  LoaderCircle,
  Play,
  RefreshCw,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import ModelViewer from "./ModelViewer";
import TrainingWorkspace from "./TrainingWorkspace";
import type { Checkpoint, GenerationJob, SystemStatus } from "./types";

const STAGES = [
  { id: "upload", label: "接收输入" },
  { id: "load", label: "加载模型" },
  { id: "preprocess", label: "处理图像" },
  { id: "inference", label: "推理体素" },
  { id: "mesh", label: "构建网格" },
  { id: "complete", label: "生成完成" },
];

function checkpointLabel(checkpoint: Checkpoint) {
  if (checkpoint.is_demo) return `${checkpoint.name} · 未训练演示`;
  return `${checkpoint.name} · ${checkpoint.epochs ?? 0} 轮`;
}

function stageState(stageId: string, job: GenerationJob | null) {
  if (!job) return "waiting";
  if (job.status === "failed") {
    const failedIndex = STAGES.findIndex((stage) => stage.id === job.failed_stage);
    const stageIndex = STAGES.findIndex((stage) => stage.id === stageId);
    if (stageIndex < failedIndex) return "done";
    if (stageIndex === failedIndex) return "failed";
    return "waiting";
  }
  const currentIndex = STAGES.findIndex((stage) => stage.id === job.stage);
  const stageIndex = STAGES.findIndex((stage) => stage.id === stageId);
  if (job.status === "completed" || stageIndex < currentIndex) return "done";
  if (stageIndex === currentIndex) return "active";
  return "waiting";
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [systemError, setSystemError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [threshold, setThreshold] = useState(0.45);
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [workspace, setWorkspace] = useState<"generate" | "train">("generate");

  const refreshSystem = async () => {
    try {
      const response = await fetch("/api/system");
      if (!response.ok) throw new Error("服务状态读取失败");
      const data = (await response.json()) as SystemStatus;
      setSystem(data);
      setSystemError(null);
      setSelectedCheckpoint((current) => {
        if (data.checkpoints.some((checkpoint) => checkpoint.name === current)) return current;
        const trained = data.checkpoints.find((checkpoint) => !checkpoint.is_demo && !checkpoint.error);
        return trained?.name ?? data.checkpoints.find((checkpoint) => !checkpoint.error)?.name ?? "";
      });
    } catch (error) {
      setSystemError(error instanceof Error ? error.message : "服务不可用");
    }
  };

  useEffect(() => {
    void refreshSystem();
    const timer = window.setInterval(() => void refreshSystem(), 8000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/jobs/${job.id}`);
      if (response.ok) setJob((await response.json()) as GenerationJob);
    }, 350);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.logs.length]);

  const setInputFile = (nextFile: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!nextFile && fileInputRef.current) fileInputRef.current.value = "";
    setFile(nextFile);
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null);
    setSubmitError(null);
    setJob(null);
  };

  const isWorking = job?.status === "queued" || job?.status === "running";
  const selected = useMemo(
    () => system?.checkpoints.find((checkpoint) => checkpoint.name === selectedCheckpoint),
    [selectedCheckpoint, system],
  );

  const startGeneration = async () => {
    if (!file || !selectedCheckpoint || isWorking) return;
    setSubmitError(null);
    setJob(null);
    const form = new FormData();
    form.append("image", file);
    form.append("checkpoint", selectedCheckpoint);
    form.append("threshold", String(threshold));
    try {
      const response = await fetch("/api/jobs", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "任务创建失败");
      setJob(data as GenerationJob);
    } catch (error) {
      const message = error instanceof Error ? error.message : "任务创建失败";
      setSubmitError(
        message === "Failed to fetch"
          ? "无法连接生成服务，请运行 start_web.ps1 后重试。"
          : message,
      );
      void refreshSystem();
    }
  };

  const download = (url: string, filename: string) => {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  };

  const checkpoints = system?.checkpoints.filter((checkpoint) => !checkpoint.error) ?? [];

  return (
    <div className={`app-shell ${workspace === "train" ? "training-mode" : "generation-mode"}`}>
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><Box size={20} /></div>
          <div>
            <h1>Voxel Studio</h1>
            <span>Image to 3D</span>
          </div>
        </div>
        <div className="workspace-switch" role="tablist" aria-label="工作模式">
          <Button
            size="sm"
            variant={workspace === "generate" ? "secondary" : "ghost"}
            aria-pressed={workspace === "generate"}
            onPress={() => setWorkspace("generate")}
          >
            <Box size={15} />生成
          </Button>
          <Button
            size="sm"
            variant={workspace === "train" ? "secondary" : "ghost"}
            aria-pressed={workspace === "train"}
            onPress={() => setWorkspace("train")}
          >
            <BrainCircuit size={15} />训练
          </Button>
        </div>
        <div className="system-state">
          <Chip color={system?.cuda_available ? "success" : "warning"} variant="soft" size="sm">
            <Cpu size={13} />
            <Chip.Label>{system?.device ?? "检测设备"}</Chip.Label>
          </Chip>
          <Chip color={checkpoints.length ? "default" : "warning"} variant="soft" size="sm">
            <FileBox size={13} />
            <Chip.Label>{checkpoints.length} 个检查点</Chip.Label>
          </Chip>
          <Button isIconOnly aria-label="刷新状态" variant="ghost" size="sm" onPress={refreshSystem}>
            <RefreshCw size={16} />
          </Button>
        </div>
      </header>

      {workspace === "train" ? <TrainingWorkspace checkpoints={checkpoints} /> : <>
      <aside className="input-panel">
        <div className="panel-heading">
          <span className="step-number">01</span>
          <div><h2>输入图像</h2><p>PNG · JPG · WEBP</p></div>
        </div>

        <input
          ref={fileInputRef}
          hidden
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          onChange={(event) => setInputFile(event.target.files?.[0] ?? null)}
        />
        <div
          className={`upload-zone ${dragActive ? "is-dragging" : ""} ${previewUrl ? "has-image" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            setInputFile(event.dataTransfer.files?.[0] ?? null);
          }}
        >
          {previewUrl ? <img src={previewUrl} alt="上传图片预览" /> : <ImageIcon size={30} strokeWidth={1.5} />}
          <div className="upload-copy">
            <strong>{file?.name ?? "选择输入图片"}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "拖放或从本机选择"}</span>
          </div>
          <div className="upload-actions">
            <Button size="sm" variant="secondary" onPress={() => fileInputRef.current?.click()}>
              <Upload size={16} />选择
            </Button>
            {file ? (
              <Button isIconOnly size="sm" variant="ghost" aria-label="移除图片" onPress={() => setInputFile(null)}>
                <X size={16} />
              </Button>
            ) : null}
          </div>
        </div>

        <div className="control-section">
          <div className="panel-heading compact">
            <span className="step-number">02</span>
            <div><h2>推理设置</h2><p>检查点与体素阈值</p></div>
          </div>
          <Select
            fullWidth
            placeholder="选择检查点"
            value={selectedCheckpoint || null}
            variant="secondary"
            onChange={(value) => setSelectedCheckpoint(value ? String(value) : "")}
          >
            <Label>模型检查点</Label>
            <Select.Trigger>
              <Select.Value />
              <Select.Indicator />
            </Select.Trigger>
            <Select.Popover>
              <ListBox>
                {checkpoints.map((checkpoint) => (
                  <ListBox.Item key={checkpoint.name} id={checkpoint.name} textValue={checkpointLabel(checkpoint)}>
                    <div className="checkpoint-option">
                      <strong>{checkpoint.name}</strong>
                      <span>{checkpoint.is_demo ? "未训练演示" : `${checkpoint.epochs ?? 0} 轮 · ${checkpoint.resolution ?? 16}³`}</span>
                    </div>
                    <ListBox.ItemIndicator />
                  </ListBox.Item>
                ))}
              </ListBox>
            </Select.Popover>
          </Select>

          <Slider
            aria-label="体素阈值"
            minValue={0.2}
            maxValue={0.7}
            step={0.01}
            value={threshold}
            onChange={(value) => setThreshold(Number(value))}
          >
            <Label>体素阈值</Label>
            <Slider.Output>{threshold.toFixed(2)}</Slider.Output>
            <Slider.Track>
              <Slider.Fill />
              <Slider.Thumb />
            </Slider.Track>
          </Slider>

          {selected?.is_demo ? (
            <div className="notice warning"><TriangleAlert size={16} /><span>当前检查点未训练，仅验证流程。</span></div>
          ) : null}
          {system?.training_active ? (
            <div className="notice warning"><TriangleAlert size={16} /><span>模型正在训练，生成任务暂时停用。</span></div>
          ) : null}
          {systemError || submitError || job?.error ? (
            <div className="notice danger"><TriangleAlert size={16} /><span>{systemError ?? submitError ?? job?.error}</span></div>
          ) : null}

          <Button
            fullWidth
            size="lg"
            isDisabled={!file || !selectedCheckpoint || isWorking || Boolean(systemError) || Boolean(system?.training_active)}
            isPending={isWorking}
            onPress={startGeneration}
          >
            {isWorking ? <LoaderCircle className="spin" size={18} /> : <Play size={18} fill="currentColor" />}
            {isWorking ? "正在生成" : "生成 3D 模型"}
          </Button>
        </div>

        {job?.status === "completed" ? (
          <div className="result-actions">
            <Button fullWidth variant="secondary" onPress={() => download(job.result_obj!, "result.obj")}>
              <Download size={16} />下载 OBJ
            </Button>
            <Button isIconOnly variant="ghost" aria-label="下载体素数据" onPress={() => download(job.result_npy!, "result.npy")}>
              <FileBox size={17} />
            </Button>
          </div>
        ) : null}
      </aside>

      <main className="model-stage">
        <div className="stage-label">
          <span>MODEL VIEWPORT</span>
          {job?.status === "completed" ? <strong>{job.vertex_count?.toLocaleString()} VERTICES</strong> : null}
        </div>
        <ModelViewer modelUrl={job?.result_obj} isWorking={Boolean(isWorking)} />
      </main>

      <aside className="process-panel">
        <div className="panel-heading">
          <span className="step-number">03</span>
          <div><h2>生成流程</h2><p>{job ? `任务 ${job.id.slice(0, 8)}` : "等待任务"}</p></div>
        </div>
        <ProgressBar aria-label="生成进度" value={job?.progress ?? 0} color={job?.status === "failed" ? "danger" : "accent"}>
          <ProgressBar.Output />
          <ProgressBar.Track><ProgressBar.Fill /></ProgressBar.Track>
        </ProgressBar>

        <ol className="stage-list">
          {STAGES.map((stage) => {
            const state = stageState(stage.id, job);
            return (
              <li key={stage.id} className={`stage-item ${state}`}>
                <span className="stage-icon">
                  {state === "done" ? <Check size={14} /> : state === "active" ? <LoaderCircle className="spin" size={14} /> : <CircleDashed size={14} />}
                </span>
                <span>{stage.label}</span>
              </li>
            );
          })}
        </ol>

        {job?.status === "completed" ? (
          <div className="result-metrics">
            <div><span>体素</span><strong>{job.voxel_count?.toLocaleString()}</strong></div>
            <div><span>三角面</span><strong>{job.triangle_count?.toLocaleString()}</strong></div>
            <div><span>用时</span><strong>{job.elapsed_seconds}s</strong></div>
          </div>
        ) : null}

        <div className="log-heading"><span>运行日志</span><span>LIVE</span></div>
        <div ref={logRef} className="job-log">
          {job?.logs.length ? job.logs.map((entry, index) => (
            <div key={`${entry.time}-${index}`}><time>{entry.time}</time><span>{entry.message}</span></div>
          )) : <div className="log-empty">尚无任务日志</div>}
        </div>
      </aside>
      </>}
    </div>
  );
}
