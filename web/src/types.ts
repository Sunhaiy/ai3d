export type Checkpoint = {
  name: string;
  size_mb: number;
  modified_at: string;
  is_demo: boolean;
  epochs?: number;
  image_size?: number;
  resolution?: number;
  latent_dim?: number;
  architecture?: string;
  target_epochs?: number;
  validation_loss?: number;
  initial_checkpoint?: string | null;
  error?: string;
};

export type SystemStatus = {
  cuda_available: boolean;
  device: string;
  checkpoints: Checkpoint[];
  training_active: boolean;
};

export type JobLog = {
  time: string;
  message: string;
};

export type GenerationJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  failed_stage?: string;
  progress: number;
  checkpoint: string;
  threshold: number;
  input_url: string;
  created_at: string;
  updated_at: string;
  logs: JobLog[];
  error: string | null;
  elapsed_seconds?: number;
  voxel_count?: number;
  vertex_count?: number;
  triangle_count?: number;
  result_obj?: string;
  result_npy?: string;
};

export type TrainingDataset = {
  ready: boolean;
  path: string;
  size_mb: number;
  mesh_count: number;
  matched_mesh_count: number;
  missing_mesh_count: number;
  ignored_mesh_count: number;
  image_count: number;
  pair_count: number;
  min_views: number;
  max_views: number;
  resolution: number;
  image_size: number;
  target_count: number;
  point_count: number;
  representation: string;
  preview_path: string;
  inspection_error?: string;
};

export type TrainingHistoryPoint = {
  epoch: number;
  train_loss: number;
  validation_loss: number | null;
  iou: number | null;
};

export type TrainingStatus = {
  status: "idle" | "preparing" | "training" | "pausing" | "paused" | "stopping" | "stopped" | "completed" | "failed";
  stage: string;
  progress: number;
  data_root: string;
  dataset: TrainingDataset;
  config: {
    epochs: number;
    batch_size: number;
    resolution: number;
    image_size: number;
    run_name: string;
    initial_checkpoint: string | null;
    max_hours: number;
    architecture: string;
  };
  current_epoch: number;
  total_epochs: number;
  current_batch: number;
  total_batches: number;
  metrics: {
    train_loss: number | null;
    validation_loss: number | null;
    iou: number | null;
  };
  history: TrainingHistoryPoint[];
  logs: JobLog[];
  gpu: {
    utilization: number;
    memory_used_mb: number;
    memory_total_mb: number;
    temperature_c: number;
  } | null;
  pid: number | null;
  started_at: string | null;
  updated_at: string;
  elapsed_seconds: number;
  error: string | null;
  can_resume: boolean;
  output_checkpoint: string | null;
};
