// frontend/src/lib/api.ts

const DEFAULT_API_BASE = import.meta.env.PROD ? "/api" : "http://localhost:8000";

const RAW_API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE;

export const API_BASE = RAW_API_BASE.replace(/\/+$/, ""); // trim trailing slashes

// -----------------------------
// JSON-safe types (no `any`)
// -----------------------------
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type ColumnDiagnostic = {
  non_null_count: number;
  unique_count: number;
  unique_ratio: number;
  mean_group_size: number;
  max_frequency_share: number;
  timestamp_parse_ratio: number;
  looks_event_unique: boolean;
  looks_timestamp_like: boolean;
};

// -----------------------------
// Types (match backend responses)
// -----------------------------
export type DatasetUploadResponse = {
  dataset_id: string;
  stored_path: string;
  raw_path?: string | null;
  preprocessed_path?: string | null;
  split_dataset_path?: string | null;
  split_paths?: Record<string, string> | null;
  split_source?: "generated" | "uploaded" | null;
  split_config?: { test_size: number; val_split: number } | null;
  is_preprocessed?: boolean;
  preprocessed_at?: string | null;
  num_events: number;
  num_cases: number;
  columns: string[];
  column_types: Record<string, "categorical" | "numerical">;
  detected_mapping: Record<string, string>;
  column_diagnostics?: Record<string, ColumnDiagnostic>;
  preview: Array<Record<string, JsonValue>>;
};

export type DatasetMeta = {
  dataset_id: string;
  stored_path: string;
  raw_path?: string | null;
  preprocessed_path?: string | null;
  split_dataset_path?: string | null;
  split_paths?: Record<string, string> | null;
  split_source?: "generated" | "uploaded" | null;
  split_config?: { test_size: number; val_split: number } | null;
  is_preprocessed?: boolean;
  preprocessed_at?: string | null;
  preprocessing_options?: Record<string, boolean> | null;
  num_events: number;
  num_cases: number;
  columns: string[];
  column_types: Record<string, "categorical" | "numerical">;
  detected_mapping: Record<string, string>;
  column_diagnostics?: Record<string, ColumnDiagnostic>;
  created_at: string;
};

export type RunStatus = {
  run_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at?: string;
  started_at?: string;
  finished_at?: string;
  pid?: number;
  error?: string | null;
};

export type RunLogsRes = {
  run_id: string;
  lines: string[];
};

export type CreateRunReq = {
  dataset_id: string;
  model_type: "gnn" | "transformer";
  task: "next_activity" | "custom_activity" | "event_time" | "remaining_time" | "unified";
  config?: Record<string, JsonValue>;
  split?: { test_size: number; val_split: number };
  explainability?: JsonValue; // e.g. "none" | null | {...}
  mapping_mode?: "auto" | "manual";
  target_column?: string | null;
  column_mapping?: {
    case_id: string;
    activity: string;
    timestamp: string;
    resource?: string | null;
  } | null;
};

export type CreateRunRes = {
  run_id: string;
  status: string; // backend returns "queued"
};

export type ArtifactsListRes = {
  run_id: string;
  artifacts: string[];
};

export type PreprocessOptions = {
  sort_and_normalize_timestamps?: boolean;
  check_millisecond_order?: boolean;
  impute_categorical?: boolean;
  impute_numeric_neighbors?: boolean;
  drop_missing_timestamps?: boolean;
  fill_remaining_missing?: boolean;
  remove_duplicates?: boolean;
};

export type SplitConfig = {
  test_size: number;
  val_split: number;
};

export type SampleDatasetInfo = {
  name: string;
  size_bytes?: number | null;
  format: "csv" | "xes";
  configured: boolean;
};

// -----------------------------
// Low-level helpers
// -----------------------------
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(url, init);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `API ${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`
    );
  }

  return res;
}

function encodeArtifactPath(path: string): string {
  // Keep slashes but encode each segment
  return path
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
}

// -----------------------------
// API functions
// -----------------------------
export async function health(): Promise<{ ok: boolean; service: string }> {
  const res = await apiFetch(`${API_BASE}/health`);
  return (await res.json()) as { ok: boolean; service: string };
}

export async function uploadDataset(
  file: File,
  opts?: { preprocessed?: boolean }
): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const url =
    opts?.preprocessed === true
      ? `${API_BASE}/datasets/upload?preprocessed=true`
      : `${API_BASE}/datasets/upload`;

  const res = await apiFetch(url, {
    method: "POST",
    body: form,
  });

  return (await res.json()) as DatasetUploadResponse;
}

export async function listSampleDatasets(): Promise<SampleDatasetInfo[]> {
  const res = await apiFetch(`${API_BASE}/sample-datasets`);
  return (await res.json()) as SampleDatasetInfo[];
}

export function sampleDatasetUrl(name: string): string {
  return `${API_BASE}/sample-datasets/${encodeURIComponent(name)}`;
}

export async function downloadSampleDataset(name: string): Promise<Blob> {
  const res = await apiFetch(sampleDatasetUrl(name));
  return res.blob();
}

export async function getDataset(dataset_id: string): Promise<DatasetMeta> {
  const res = await apiFetch(
    `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}`
  );
  return (await res.json()) as DatasetMeta;
}

export async function preprocessDataset(
  dataset_id: string,
  options: PreprocessOptions
): Promise<DatasetUploadResponse> {
  const res = await apiFetch(
    `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/preprocess`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options ?? {}),
    }
  );
  return (await res.json()) as DatasetUploadResponse;
}

export function preprocessedDatasetUrl(dataset_id: string): string {
  return `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/preprocessed`;
}

export async function generateSplits(
  dataset_id: string,
  config: SplitConfig
): Promise<DatasetUploadResponse> {
  const res = await apiFetch(
    `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/splits/generate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }
  );
  return (await res.json()) as DatasetUploadResponse;
}

export async function uploadSplits(
  dataset_id: string,
  train: File,
  val: File,
  test: File
): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("train", train);
  form.append("val", val);
  form.append("test", test);

  const res = await apiFetch(
    `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/splits/upload`,
    {
      method: "POST",
      body: form,
    }
  );
  return (await res.json()) as DatasetUploadResponse;
}

export async function uploadSplitsNewDataset(
  train: File,
  val: File,
  test: File
): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("train", train);
  form.append("val", val);
  form.append("test", test);

  const res = await apiFetch(`${API_BASE}/datasets/splits/upload`, {
    method: "POST",
    body: form,
  });
  return (await res.json()) as DatasetUploadResponse;
}

export function splitDownloadUrl(dataset_id: string, split: "train" | "val" | "test"): string {
  return `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/splits/${split}`;
}

export async function createRun(body: CreateRunReq): Promise<CreateRunRes> {
  const res = await apiFetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...body,
      config: body.config ?? {},
      split: body.split ?? { test_size: 0.2, val_split: 0.5 },
    }),
  });

  return (await res.json()) as CreateRunRes;
}

export async function getRun(run_id: string): Promise<RunStatus> {
  const res = await apiFetch(`${API_BASE}/runs/${encodeURIComponent(run_id)}`);
  return (await res.json()) as RunStatus;
}

export async function getRunLogs(run_id: string, tail = 50): Promise<RunLogsRes> {
  const res = await apiFetch(
    `${API_BASE}/runs/${encodeURIComponent(run_id)}/logs?tail=${tail}`
  );
  return (await res.json()) as RunLogsRes;
}

export function runLogsTextUrl(run_id: string): string {
  return `${API_BASE}/runs/${encodeURIComponent(run_id)}/logs.txt`;
}

export async function pollRunUntilDone(
  run_id: string,
  opts?: { intervalMs?: number; timeoutMs?: number }
): Promise<RunStatus> {
  const intervalMs = opts?.intervalMs ?? 1000;
  const timeoutMs = opts?.timeoutMs ?? 10 * 60 * 1000;

  const start = Date.now();

  while (true) {
    const status = await getRun(run_id);
    if (status.status === "succeeded" || status.status === "failed") return status;

    if (Date.now() - start > timeoutMs) {
      throw new Error(`Polling timed out after ${timeoutMs}ms for run ${run_id}`);
    }

    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function listArtifacts(run_id: string): Promise<ArtifactsListRes> {
  const res = await apiFetch(
    `${API_BASE}/runs/${encodeURIComponent(run_id)}/artifacts`
  );
  return (await res.json()) as ArtifactsListRes;
}

export function artifactUrl(run_id: string, artifact_path: string): string {
  const rid = encodeURIComponent(run_id);
  const ap = encodeArtifactPath(artifact_path);
  return `${API_BASE}/runs/${rid}/artifacts/${ap}`;
}

export function artifactsZipUrl(run_id: string): string {
  const rid = encodeURIComponent(run_id);
  return `${API_BASE}/runs/${rid}/artifacts.zip`;
}

export async function fetchArtifactBlob(
  run_id: string,
  artifact_path: string
): Promise<Blob> {
  const res = await apiFetch(artifactUrl(run_id, artifact_path));
  return await res.blob();
}

export async function fetchArtifactJson<T extends JsonValue = JsonValue>(
  run_id: string,
  artifact_path: string
): Promise<T> {
  const res = await apiFetch(artifactUrl(run_id, artifact_path));
  return (await res.json()) as T;
}
