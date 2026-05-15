// frontend/src/components/steps/Step6Review.tsx
import { useEffect, useRef, useState } from "react";

import type { DatasetUploadResponse, RunStatus } from "../../lib/api";
import { artifactUrl, runLogsTextUrl } from "../../lib/api";

type ManualMapping = {
  case_id: string;
  activity: string;
  timestamp: string;
  resource: string | null;
};

function formatPredictionTask(task: string | null): string {
  if (!task) return "-";
  const s = task.toLowerCase().trim();
  if (s === "next_activity" || s.includes("next activity")) return "Next Activity Prediction";
  if (s === "custom_activity" || s.includes("custom")) return "Custom Activity Prediction";
  if (s === "event_time" || s.includes("event time") || s === "timestamp") return "Event Time Prediction";
  if (s === "remaining_time" || s.includes("remaining time")) return "Remaining Time Prediction";
  if (s === "unified") return "Unified Prediction";
  return task;
}

function formatExplainability(v: string | null): string {
  if (!v) return "-";
  const s = v.toLowerCase().trim();
  if (s === "none") return "None";
  if (s === "all") return "Both";
  if (s === "lime") return "LIME / GraphLIME";
  if (s === "shap") return "SHAP";
  if (s === "gradient") return "Gradient-Based";
  return v;
}

type Step6ReviewProps = {
  uploadedFile: File | null;
  dataset: DatasetUploadResponse | null;

  modelType: string | null;
  predictionTask: string | null;
  explainMethod: string | null;
  mappingMode: "manual" | null;
  manualMapping: ManualMapping;
  configMode: "default" | "custom" | null;

  pipelineStatus: "idle" | "running" | "completed";
  progress: number;

  runId: string | null;
  runStatus: RunStatus | null;
  artifacts: string[];
  logs: string[];

  error: string | null;

  onStartPipeline: () => void;
  onViewResults: () => void;
};

export default function Step6Review({
  uploadedFile,
  dataset,
  modelType,
  predictionTask,
  explainMethod,
  mappingMode,
  manualMapping,
  configMode,
  pipelineStatus,
  progress,
  runId,
  runStatus,
  artifacts,
  logs,
  error,
  onStartPipeline,
  onViewResults,
}: Step6ReviewProps) {
  const [renderedAt, setRenderedAt] = useState(() => Date.now());
  const terminalRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (pipelineStatus !== "running") return;
    const interval = window.setInterval(() => {
      setRenderedAt(Date.now());
    }, 30000);
    return () => window.clearInterval(interval);
  }, [pipelineStatus]);

  useEffect(() => {
    if (!terminalRef.current) return;
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [logs]);

  const datasetName = uploadedFile?.name ?? "-";
  const datasetId = dataset?.dataset_id ?? "-";
  const events = dataset?.num_events ?? null;
  const cases = dataset?.num_cases ?? null;

  const configLabel =
    configMode === "default"
      ? "Default Configuration"
      : configMode === "custom"
      ? "Custom Configuration"
      : "-";

  const backendStatus = runStatus?.status ?? (pipelineStatus === "idle" ? "-" : "queued");
  const lastLogLine = [...logs].reverse().find((l) => l.trim().length > 0) ?? "-";
  const etaMinutes = (() => {
    if (pipelineStatus !== "running") return null;
    if (progress <= 1 || progress >= 100) return null;
    const startedAt = runStatus?.started_at ?? runStatus?.created_at ?? null;
    if (!startedAt) return null;
    const started = Date.parse(startedAt);
    if (!Number.isFinite(started)) return null;
    const elapsedMs = renderedAt - started;
    if (elapsedMs <= 0) return null;
    const remainingMs = (elapsedMs * (100 - progress)) / progress;
    return Math.max(1, Math.ceil(remainingMs / 60000));
  })();
  const mappingLabel =
    mappingMode === "manual"
      ? `Manual (case_id=${manualMapping.case_id}, activity=${manualMapping.activity}, timestamp=${manualMapping.timestamp}${
          manualMapping.resource ? `, resource=${manualMapping.resource}` : ""
        })`
      : "-";

  return (
    <div className="space-y-6 w-full">
      {/* SUMMARY */}
      <div className="border border-brand-100 rounded-xl p-6 bg-white">
        <h2 className="text-lg font-semibold mb-4">Review & Run Pipeline</h2>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <SummaryItem label="Dataset" value={datasetName} />
          <SummaryItem label="Dataset ID" value={datasetId} />
          <SummaryItem
            label="Events / Cases"
            value={
              events !== null && cases !== null
                ? `${events.toLocaleString()} events, ${cases.toLocaleString()} cases`
                : "-"
            }
          />
          <SummaryItem label="Model Type" value={modelType ?? "-"} />
          <SummaryItem label="Prediction Task" value={formatPredictionTask(predictionTask)} />
          <SummaryItem label="Explainability" value={formatExplainability(explainMethod)} />
          <SummaryItem label="Column Mapping" value={mappingLabel} />
          <SummaryItem label="Configuration" value={configLabel} />
          <SummaryItem label="Run ID" value={runId ?? "-"} />
          <SummaryItem label="Run Status" value={backendStatus} />
        </div>

        {error && (
          <div className="mt-4 border border-red-200 bg-red-50 rounded-lg p-4 text-sm text-red-700">
            <div className="font-medium mb-1">Run failed</div>
            <div className="break-words">{error}</div>
          </div>
        )}
      </div>

      {/* READY */}
      {pipelineStatus === "idle" && (
        <div className="border border-brand-200 rounded-xl p-6 bg-brand-50">
          <p className="text-sm text-gray-700 mb-4">
            Your pipeline is configured and ready to run.
          </p>

          <button
            className="px-5 py-2 bg-brand-600 text-white rounded hover:bg-brand-700"
            onClick={onStartPipeline}
            disabled={!dataset}
            title={!dataset ? "Upload a dataset first" : undefined}
          >
            Start Pipeline Execution
          </button>
        </div>
      )}

      {/* RUNNING */}
      {pipelineStatus === "running" && (
        <div className="border border-brand-100 rounded-xl p-6 bg-white">
          <div className="mb-3 text-sm font-medium">Pipeline Execution in Progress</div>

          <div className="w-full bg-brand-100 rounded-full h-2 mb-2">
            <div
              className="h-2 rounded-full transition-all bg-gradient-to-r from-brand-600 to-accent-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="text-sm text-gray-600">
            {progress}% - backend status: <span className="font-medium">{backendStatus}</span>
          </div>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-gray-700">
            <div>
              <div className="font-medium">{lastLogLine}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Estimated time remaining</div>
              <div className="font-medium">{etaMinutes ? `${etaMinutes} min` : "-"}</div>
            </div>
          </div>

        </div>
      )}

      {runId && (
        <div className="border border-brand-100 rounded-xl p-6 bg-white">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">Live Terminal</div>
              <div className="text-xs text-gray-500">
                Streaming the latest backend logs for this run.
              </div>
            </div>
            <a
              href={runLogsTextUrl(runId)}
              className="rounded-md border border-brand-200 bg-white px-3 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
              download
            >
              Download logs
            </a>
          </div>

          <pre
            ref={terminalRef}
            className="max-h-80 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-4 font-mono text-xs leading-6 text-green-100"
          >
            {logs.length > 0 ? logs.join("\n") : "Waiting for backend logs..."}
          </pre>
        </div>
      )}

      {/* COMPLETED */}
      {pipelineStatus === "completed" && (
        <div className="border rounded-xl p-6 bg-green-50">
          <div className="text-green-700 font-medium mb-3">
            Pipeline execution completed successfully.
          </div>

          {runId && artifacts.length > 0 && (
            <div className="mb-4">
              <div className="text-sm font-medium text-gray-800 mb-2">Artifacts</div>
              <ul className="text-sm list-disc ml-5 space-y-1">
                {artifacts.map((a) => (
                  <li key={a}>
                    <a
                      className="text-brand-700 hover:underline"
                      href={artifactUrl(runId, a)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {a}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            className="px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            onClick={onViewResults}
          >
            View Results
          </button>
        </div>
      )}
    </div>
  );
}

/* ----------------- Helper ----------------- */

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-brand-600 text-xs mb-1">{label}</div>
      <div className="font-medium text-brand-900 break-words">{value}</div>
    </div>
  );
}

