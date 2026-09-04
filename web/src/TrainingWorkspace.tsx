import {
  Button,
  Chip,
  Input,
  Label,
  ListBox,
  NumberField,
  ProgressBar,
  Select,
  TextField,
} from "@heroui/react";
import {
  Activity,
  Box,
  Check,
  Database,
  FolderSearch,
  Gauge,
  HardDrive,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Square,
  Thermometer,
  Timer,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { Checkpoint, TrainingHistoryPoint, TrainingStatus } from "./types";

const ACTIVE_STATUSES = new Set(["preparing", "training", "pausing", "stopping"]);
const RESOLUTION_OPTIONS = [
  { id: "16", label: "16³ · 快速", batchSize: 32, epochs: 100, maxHours: 0 },
  { id: "32", label: "32³ · 推荐", batchSize: 32, epochs: 100, maxHours: 0 },
  { id: "64", label: "64³ · 精细", batchSize: 16, epochs: 100, maxHours: 0 },
  { id: "128", label: "128³ · 实验", batchSize: 4, epochs: 100, maxHours: 0 },
  { id: "256", label: "256³ · 长时训练", batchSize: 1, epochs: 1000, maxHours: 72 },
  { id: "512", label: "512³ · 隐式表面", batchSize: 1, epochs: 1000, maxHours: 72 },
  { id: "1024", label: "1024³ · 隐式表面", batchSize: 1, epochs: 1000, maxHours: 72 },
];
const SCRATCH_MODEL = "__scratch__";

const STATUS_LABELS: Record<TrainingStatus["status"], string> = {
  idle: "等待开始",
  preparing: "正在制作数据",
  training: "正在训练",
  pausing: "正在暂停",
  paused: "已暂停",
  stopping: "正在停止",
  stopped: "已停止",
  completed: "已完成",
  failed: "任务失败",
};

function formatDuration(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = whole % 60;
  return hours > 0
    ? `${hours}时 ${String(minutes).padStart(2, "0")}分`
    : `${minutes}分 ${String(rest).padStart(2, "0")}秒`;
}

function metric(value: number | null, digits = 4) {
  return value === null ? "--" : value.toFixed(digits);
}

function MiniChart({
  title,
  values,
  color,
}: {
  title: string;
  values: Array<{ epoch: number; value: number }>;
  color: string;
}) {
  const chart = useMemo(() => {
    if (!values.length) return null;
    const width = 360;
    const height = 116;
    const inset = 8;
    const rawMin = Math.min(...values.map((point) => point.value));
    const rawMax = Math.max(...values.map((point) => point.value));
    const padding = Math.max((rawMax - rawMin) * 0.12, 0.01);
    const min = Math.max(0, rawMin - padding);
    const max = rawMax + padding;
    const range = Math.max(max - min, 0.001);
    const points = values.map((point, index) => {
      const x = values.length === 1
        ? width / 2
        : inset + (index / (values.length - 1)) * (width - inset * 2);
      const y = inset + (1 - (point.value - min) / range) * (height - inset * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    return { width, height, min, max, points };
  }, [values]);

  return (
    <div className="metric-chart">
      <div className="metric-chart-heading">
        <span>{title}</span>
        <strong>{values.length ? values.at(-1)!.value.toFixed(4) : "--"}</strong>
      </div>
      {chart ? (
        <div className="chart-canvas">
          <span className="chart-max">{chart.max.toFixed(3)}</span>
          <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label={`${title}趋势`}>
            <line x1="0" y1="58" x2="360" y2="58" className="chart-grid-line" />
            <polyline points={chart.points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
          </svg>
          <span className="chart-min">{chart.min.toFixed(3)}</span>
        </div>
      ) : <div className="chart-empty">等待首轮指标</div>}
    </div>
  );
}

async function postTrainingAction(url: string, body?: object) {
  const response = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ?? "操作失败");
  return data as TrainingStatus;
}

export default function TrainingWorkspace({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const logRef = useRef<HTMLDivElement>(null);
  const initializedRef = useRef(false);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [dataRoot, setDataRoot] = useState("E:\\modeldata");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [maxHours, setMaxHours] = useState(0);
  const [resolution, setResolution] = useState(32);
  const [runName, setRunName] = useState("");
  const [initialCheckpoint, setInitialCheckpoint] = useState(SCRATCH_MODEL);
  const [pendingAction, setPendingAction] = useState<"prepare" | "train" | "pause" | "resume" | "stop" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const response = await fetch("/api/training");
      if (!response.ok) throw new Error("训练状态读取失败");
      const next = (await response.json()) as TrainingStatus;
      setStatus(next);
      setConnectionError(null);
      if (!initializedRef.current) {
        const useDatasetResolution = next.status === "idle" && next.dataset.ready;
        const initialResolution = useDatasetResolution
          ? next.dataset.resolution
          : next.config.resolution || 32;
        const resolutionOption = RESOLUTION_OPTIONS.find(
          (option) => Number(option.id) === initialResolution,
        );
        setDataRoot(next.data_root);
        setEpochs(next.config.epochs);
        setBatchSize(useDatasetResolution
          ? resolutionOption?.batchSize ?? next.config.batch_size
          : next.config.batch_size);
        setMaxHours(useDatasetResolution
          ? resolutionOption?.maxHours ?? next.config.max_hours ?? 0
          : next.config.max_hours ?? 0);
        setResolution(initialResolution);
        setRunName(next.config.run_name ?? "");
        setInitialCheckpoint(next.config.initial_checkpoint ?? SCRATCH_MODEL);
        initializedRef.current = true;
      }
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "训练服务不可用");
    }
  };

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status?.logs.length]);

  const runAction = async (
    action: "prepare" | "train" | "pause" | "resume" | "stop",
    url: string,
    body?: object,
  ) => {
    setPendingAction(action);
    setActionError(null);
    try {
      const next = await postTrainingAction(url, body);
      setStatus(next);
      if (action === "train") setRunName(next.config.run_name ?? "");
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "操作失败");
    } finally {
      setPendingAction(null);
    }
  };

  const active = status ? ACTIVE_STATUSES.has(status.status) : false;
  const preparing = status?.status === "preparing";
  const training = status?.status === "training";
  const pausing = status?.status === "pausing";
  const paused = status?.status === "paused";
  const canResume = status?.can_resume ?? false;
  const dataset = status?.dataset;
  const datasetMatches = Boolean(
    dataset?.ready && dataset.resolution === resolution && dataset.image_size === 128,
  );
  const gpu = status?.gpu;
  const selectedInitialCheckpoint = checkpoints.find(
    (checkpoint) => checkpoint.name === initialCheckpoint,
  );
  const initialCheckpointMatches = initialCheckpoint === SCRATCH_MODEL || Boolean(
    selectedInitialCheckpoint,
  );
  const targetArchitecture = resolution >= 512
    ? "implicit"
    : resolution >= 256
      ? "scalable"
      : "legacy";
  const lossValues = (status?.history ?? [])
    .filter((point): point is TrainingHistoryPoint & { validation_loss: number } => point.validation_loss !== null)
    .map((point) => ({ epoch: point.epoch, value: point.validation_loss }));
  const iouValues = (status?.history ?? [])
    .filter((point): point is TrainingHistoryPoint & { iou: number } => point.iou !== null)
    .map((point) => ({ epoch: point.epoch, value: point.iou }));

  const prepareState = datasetMatches ? "done" : preparing ? "active" : "waiting";
  const trainState = status?.stage === "checkpoint_ready"
    ? "done"
    : training || pausing || paused || status?.status === "stopping"
      ? "active"
      : "waiting";

  return (
    <section className="training-workspace">
      <aside className="training-controls">
        <div className="training-title-block">
          <span className="section-kicker">TRAINING CONTROL</span>
          <h2>训练控制</h2>
          <Chip
            size="sm"
            variant="soft"
            color={status?.status === "failed" ? "danger" : active ? "warning" : dataset?.ready ? "success" : "default"}
          >
            <Activity size={13} />
            <Chip.Label>{status ? STATUS_LABELS[status.status] : "连接中"}</Chip.Label>
          </Chip>
        </div>

        <div className="training-section">
          <div className="operation-heading">
            <span className={`operation-state ${prepareState}`}>
              {prepareState === "done" ? <Check size={14} /> : preparing ? <LoaderCircle className="spin" size={14} /> : "1"}
            </span>
            <div><h3>制作训练集</h3><span>图片与 3D 配对体素化</span></div>
          </div>
          <TextField fullWidth value={dataRoot} onChange={setDataRoot} isDisabled={active || canResume}>
            <Label>数据根目录</Label>
            <Input variant="secondary" />
          </TextField>
          <Select
            fullWidth
            value={String(resolution)}
            variant="secondary"
            isDisabled={active || canResume}
            onChange={(value) => {
              const next = Number(value);
              const option = RESOLUTION_OPTIONS.find((item) => Number(item.id) === next);
              setResolution(next);
              if (option) {
                setBatchSize(option.batchSize);
                setEpochs(option.epochs);
                setMaxHours(option.maxHours);
              }
            }}
          >
            <Label>体素分辨率</Label>
            <Select.Trigger>
              <Select.Value />
              <Select.Indicator />
            </Select.Trigger>
            <Select.Popover>
              <ListBox>
                {RESOLUTION_OPTIONS.map((option) => (
                  <ListBox.Item key={option.id} id={option.id} textValue={option.label}>
                    {option.label}
                    <ListBox.ItemIndicator />
                  </ListBox.Item>
                ))}
              </ListBox>
            </Select.Popover>
          </Select>
          <div className="dataset-summary">
            <div><span>模型</span><strong>{dataset?.matched_mesh_count.toLocaleString() ?? "--"}</strong></div>
            <div><span>图片</span><strong>{dataset?.image_count.toLocaleString() ?? "--"}</strong></div>
            <div><span>训练对</span><strong>{dataset?.pair_count.toLocaleString() ?? "--"}</strong></div>
            <div><span>当前体素</span><strong>{dataset?.resolution ? `${dataset.resolution}³` : "--"}</strong></div>
          </div>
          {dataset?.ready ? (
            <div className="dataset-location">
              <Database size={15} />
              <div><span>处理后训练集</span><code>{dataset.path}</code></div>
            </div>
          ) : null}
          {dataset?.ready && !datasetMatches ? (
            <div className="notice warning">
              <TriangleAlert size={16} />
              <span>当前数据集为 {dataset.resolution}³，需要重新制作 {resolution}³ 数据集。</span>
            </div>
          ) : null}
          {dataset?.inspection_error ? (
            <div className="notice danger"><TriangleAlert size={16} /><span>{dataset.inspection_error}</span></div>
          ) : null}
          <Button
            fullWidth
            size="lg"
            variant={datasetMatches ? "secondary" : "primary"}
            isDisabled={active || canResume || !status}
            isPending={pendingAction === "prepare"}
            onPress={() => void runAction("prepare", "/api/training/prepare", {
              data_root: dataRoot,
              resolution,
              image_size: 128,
            })}
          >
            {preparing ? <LoaderCircle className="spin" size={18} /> : <FolderSearch size={18} />}
            {datasetMatches ? `重新制作 ${resolution}³ 训练集` : `制作 ${resolution}³ 训练集`}
          </Button>
        </div>

        <div className="training-section">
          <div className="operation-heading">
            <span className={`operation-state ${trainState}`}>
              {trainState === "done" ? <Check size={14} /> : training ? <LoaderCircle className="spin" size={14} /> : "2"}
            </span>
            <div><h3>训练模型</h3><span>CUDA · 最佳验证损失</span></div>
          </div>
          <Select
            fullWidth
            value={initialCheckpoint}
            variant="secondary"
            isDisabled={active || canResume}
            onChange={(value) => setInitialCheckpoint(String(value ?? SCRATCH_MODEL))}
          >
            <Label>初始模型</Label>
            <Select.Trigger>
              <Select.Value />
              <Select.Indicator />
            </Select.Trigger>
            <Select.Popover>
              <ListBox>
                <ListBox.Item id={SCRATCH_MODEL} textValue="从零开始 · 随机权重">
                  <div className="checkpoint-option">
                    <strong>从零开始</strong>
                    <span>随机权重</span>
                  </div>
                  <ListBox.ItemIndicator />
                </ListBox.Item>
                {checkpoints.map((checkpoint) => (
                  <ListBox.Item key={checkpoint.name} id={checkpoint.name} textValue={checkpoint.name}>
                    <div className="checkpoint-option">
                      <strong>{checkpoint.name}</strong>
                      <span>
                        {checkpoint.epochs ?? 0} 轮 · {checkpoint.resolution ?? "--"}³ · {
                          checkpoint.image_size === 128
                          && checkpoint.resolution === resolution
                          && checkpoint.latent_dim === (resolution <= 16 ? 128 : 256)
                          && (checkpoint.architecture ?? "legacy") === targetArchitecture
                            ? "完整权重"
                            : "迁移兼容层"
                        }
                      </span>
                    </div>
                    <ListBox.ItemIndicator />
                  </ListBox.Item>
                ))}
              </ListBox>
            </Select.Popover>
          </Select>
          <TextField
            fullWidth
            value={runName}
            onChange={setRunName}
            isDisabled={active || canResume}
          >
            <Label>训练成果名称</Label>
            <Input variant="secondary" placeholder="留空则自动使用当前时间" />
          </TextField>
          <div className="training-parameters">
            <NumberField
              value={epochs}
              minValue={1}
              maxValue={1000}
              step={1}
              variant="secondary"
              isDisabled={active || canResume}
              onChange={(value) => setEpochs(value ?? 100)}
            >
              <Label>训练轮数</Label>
              <NumberField.Group>
                <NumberField.DecrementButton />
                <NumberField.Input />
                <NumberField.IncrementButton />
              </NumberField.Group>
            </NumberField>
            <NumberField
              value={batchSize}
              minValue={1}
              maxValue={128}
              step={1}
              variant="secondary"
              isDisabled={active || canResume}
              onChange={(value) => setBatchSize(value ?? 32)}
            >
              <Label>批大小（每步图片数）</Label>
              <NumberField.Group>
                <NumberField.DecrementButton />
                <NumberField.Input />
                <NumberField.IncrementButton />
              </NumberField.Group>
            </NumberField>
            <NumberField
              className="training-duration-field"
              value={maxHours}
              minValue={0}
              maxValue={720}
              step={1}
              variant="secondary"
              isDisabled={active || canResume}
              onChange={(value) => setMaxHours(value ?? 0)}
            >
              <Label>最长训练时长（小时，0 不限）</Label>
              <NumberField.Group>
                <NumberField.DecrementButton />
                <NumberField.Input />
                <NumberField.IncrementButton />
              </NumberField.Group>
            </NumberField>
          </div>
          <Button
            fullWidth
            size="lg"
            isDisabled={!datasetMatches || !initialCheckpointMatches || active || !status}
            isPending={pendingAction === "train" || pendingAction === "resume"}
            onPress={() => canResume
              ? void runAction("resume", "/api/training/resume")
              : void runAction("train", "/api/training/start", {
                  epochs,
                  batch_size: batchSize,
                  max_hours: maxHours,
                  name: runName.trim() || null,
                  initial_checkpoint: initialCheckpoint === SCRATCH_MODEL ? null : initialCheckpoint,
                })}
          >
            {canResume ? <RotateCcw size={18} /> : <Play size={18} fill="currentColor" />}
            {canResume ? "继续训练" : initialCheckpoint === SCRATCH_MODEL ? "开始训练" : "基于所选模型训练"}
          </Button>
          {status?.output_checkpoint ? (
            <div className="training-output-name">
              <span>保存文件</span><strong>{status.output_checkpoint}</strong>
            </div>
          ) : null}
          {training || pausing ? (
            <Button
              fullWidth
              variant="secondary"
              isDisabled={pausing}
              isPending={pendingAction === "pause" || pausing}
              onPress={() => void runAction("pause", "/api/training/pause")}
            >
              <Pause size={16} fill="currentColor" />
              {pausing ? "正在保存暂停点" : "暂停训练"}
            </Button>
          ) : null}
          {active || canResume ? (
            <Button
              fullWidth
              variant="danger"
              isPending={pendingAction === "stop" || status?.status === "stopping"}
              onPress={() => void runAction("stop", "/api/training/stop")}
            >
              <Square size={15} fill="currentColor" />
              {canResume ? "结束本次训练" : "停止当前任务"}
            </Button>
          ) : null}
          {actionError || connectionError || status?.error ? (
            <div className="notice danger"><TriangleAlert size={16} /><span>{actionError ?? connectionError ?? status?.error}</span></div>
          ) : null}
        </div>
      </aside>

      <main className="training-dashboard">
        <div className="training-dashboard-header">
          <div>
            <span className="section-kicker">LIVE METRICS</span>
            <h2>{preparing ? "正在构建训练数据" : training || pausing ? "模型训练监控" : paused ? "训练已暂停" : "训练概览"}</h2>
          </div>
          <div className="elapsed-time"><Timer size={15} /><span>{formatDuration(status?.elapsed_seconds ?? 0)}</span></div>
        </div>

        <div className="training-progress-block">
          <div className="progress-copy">
            <span>{preparing ? "模型体素化" : training || pausing || paused ? `第 ${status?.current_epoch ?? 0} / ${status?.total_epochs ?? epochs} 轮` : STATUS_LABELS[status?.status ?? "idle"]}</span>
            <strong>{(status?.progress ?? 0).toFixed(1)}%</strong>
          </div>
          <ProgressBar aria-label="训练总进度" value={status?.progress ?? 0} color={status?.status === "failed" ? "danger" : "accent"}>
            <ProgressBar.Track><ProgressBar.Fill /></ProgressBar.Track>
          </ProgressBar>
          <div className="batch-copy">
            <span>{status?.total_batches ? `${status.current_batch} / ${status.total_batches}` : "等待任务"}</span>
            <span>PID {status?.pid ?? "--"}</span>
          </div>
        </div>

        <div className="primary-metrics">
          <div><span>训练 LOSS</span><strong>{metric(status?.metrics.train_loss ?? null)}</strong></div>
          <div><span>验证 LOSS</span><strong>{metric(status?.metrics.validation_loss ?? null)}</strong></div>
          <div><span>验证 IoU</span><strong>{metric(status?.metrics.iou ?? null)}</strong></div>
          <div><span>已完成轮数</span><strong>{status?.history.length ?? 0}</strong></div>
        </div>

        <div className="charts-grid">
          <MiniChart title="验证 LOSS" values={lossValues} color="#d08a31" />
          <MiniChart title="验证 IoU" values={iouValues} color="#43ad70" />
        </div>

        <div className="gpu-panel">
          <div className="gpu-panel-heading">
            <div><Gauge size={17} /><strong>RTX 4060</strong></div>
            <span>{gpu ? `${gpu.temperature_c}°C` : "等待 GPU"}</span>
          </div>
          <div className="gpu-meters">
            <ProgressBar aria-label="GPU 利用率" value={gpu?.utilization ?? 0} color="accent">
              <Label><Activity size={13} />GPU 利用率</Label>
              <ProgressBar.Output>{gpu ? `${gpu.utilization}%` : "--"}</ProgressBar.Output>
              <ProgressBar.Track><ProgressBar.Fill /></ProgressBar.Track>
            </ProgressBar>
            <ProgressBar
              aria-label="显存占用"
              value={gpu ? gpu.memory_used_mb / gpu.memory_total_mb * 100 : 0}
              color="warning"
            >
              <Label><HardDrive size={13} />显存</Label>
              <ProgressBar.Output>{gpu ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB` : "--"}</ProgressBar.Output>
              <ProgressBar.Track><ProgressBar.Fill /></ProgressBar.Track>
            </ProgressBar>
          </div>
          <div className="gpu-facts">
            <span><Thermometer size={13} />{gpu ? `${gpu.temperature_c}°C` : "--"}</span>
            <span><Box size={13} />{dataset?.ready ? `${dataset.resolution}³ · ${dataset.size_mb} MB` : "数据集未生成"}</span>
            <span><Database size={13} />{dataset?.pair_count.toLocaleString() ?? "--"} 对</span>
          </div>
        </div>
      </main>

      <aside className="training-log-panel">
        <div className="training-log-heading">
          <div><span className="section-kicker">PROCESS OUTPUT</span><h2>实时日志</h2></div>
          <span className={`live-indicator ${active ? "active" : ""}`}>{active ? "LIVE" : paused ? "PAUSED" : "IDLE"}</span>
        </div>
        <div ref={logRef} className="training-log">
          {status?.logs.length ? status.logs.map((entry, index) => (
            <div key={`${entry.time}-${index}`}>
              <time>{entry.time}</time>
              <span>{entry.message}</span>
            </div>
          )) : (
            <div className="training-log-empty">
              <Database size={22} />
              <span>尚无训练日志</span>
            </div>
          )}
        </div>
      </aside>
    </section>
  );
}
