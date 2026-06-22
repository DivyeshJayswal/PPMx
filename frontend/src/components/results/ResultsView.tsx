import { useEffect, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import { X } from "lucide-react";

import { artifactUrl, artifactsZipUrl, listArtifacts } from "../../lib/api";

type ResultsViewProps = {
  runId: string | null;
  onBackToPipeline: () => void;
  onRerun: () => void;
  onFinish: () => void;
};

export default function ResultsView({ runId, onBackToPipeline, onRerun, onFinish }: ResultsViewProps) {
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [pngs, setPngs] = useState<string[]>([]);
  const [summaryData, setSummaryData] = useState<Record<string, unknown> | null>(null);
  const [metricsData, setMetricsData] = useState<Record<string, unknown> | null>(null);
  const [datasetSummaryData, setDatasetSummaryData] = useState<Record<string, unknown> | null>(null);
  const [trainingSummaryData, setTrainingSummaryData] = useState<Record<string, unknown> | null>(null);
  const [evaluationRows, setEvaluationRows] = useState<Array<Record<string, string>>>([]);
  const [evaluationHeaders, setEvaluationHeaders] = useState<string[]>([]);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeImage, setActiveImage] = useState<{ src: string; name: string } | null>(
    null
  );
  const [zoom, setZoom] = useState(1);
  const pinchState = useRef<{
    active: boolean;
    startDistance: number;
    startZoom: number;
    pointers: Map<number, { x: number; y: number }>;
  }>({
    active: false,
    startDistance: 0,
    startZoom: 1,
    pointers: new Map(),
  });

  useEffect(() => {
    let cancelled = false;

    const loadArtifacts = async () => {
      if (!runId) {
        setArtifacts([]);
        setPngs([]);
        setSummaryData(null);
        setMetricsData(null);
        setDatasetSummaryData(null);
        setTrainingSummaryData(null);
        setEvaluationRows([]);
        setEvaluationHeaders([]);
        return;
      }
      setLoading(true);
      setError(null);
      setEvaluationError(null);
      try {
        const res = await listArtifacts(runId);
        if (cancelled) return;
        const artifactPaths = res.artifacts ?? [];
        const images = artifactPaths.filter((path) =>
          path.toLowerCase().endsWith(".png")
        );
        setArtifacts(artifactPaths);
        setPngs(images);
        const [summaryRes, metricsRes, datasetSummaryRes, trainingSummaryRes] = await Promise.all([
          fetch(artifactUrl(runId, "summary.json")),
          fetch(artifactUrl(runId, "metrics.json")),
          fetch(artifactUrl(runId, "dataset_summary.json")),
          fetch(artifactUrl(runId, "training_summary.json")),
        ]);
        if (cancelled) return;
        if (summaryRes.ok) {
          setSummaryData((await summaryRes.json()) as Record<string, unknown>);
        } else {
          setSummaryData(null);
        }
        if (metricsRes.ok) {
          setMetricsData((await metricsRes.json()) as Record<string, unknown>);
        } else {
          setMetricsData(null);
        }
        if (datasetSummaryRes.ok) {
          setDatasetSummaryData((await datasetSummaryRes.json()) as Record<string, unknown>);
        } else {
          setDatasetSummaryData(null);
        }
        if (trainingSummaryRes.ok) {
          setTrainingSummaryData((await trainingSummaryRes.json()) as Record<string, unknown>);
        } else {
          setTrainingSummaryData(null);
        }
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setArtifacts([]);
        setPngs([]);
        setSummaryData(null);
        setMetricsData(null);
        setDatasetSummaryData(null);
        setTrainingSummaryData(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    loadArtifacts();

    return () => {
      cancelled = true;
    };
  }, [runId]);

  useEffect(() => {
    let cancelled = false;
    const loadEvaluation = async () => {
      if (!runId) {
        setEvaluationRows([]);
        setEvaluationHeaders([]);
        return;
      }
      setEvaluationError(null);
      setEvaluationRows([]);
      setEvaluationHeaders([]);
      try {
        const discoveredEvaluationPaths = artifacts
          .filter((path) => {
            const normalized = normalizePath(path);
            return (
              normalized.includes("/evaluation/") &&
              normalized.endsWith(".csv") &&
              (normalized.endsWith("evaluation_summary.csv") ||
                normalized.endsWith("_summary.csv"))
            );
          })
          .sort((a, b) => {
            const aName = normalizePath(a);
            const bName = normalizePath(b);
            if (aName.endsWith("evaluation_summary.csv")) return -1;
            if (bName.endsWith("evaluation_summary.csv")) return 1;
            return aName.localeCompare(bName);
          });

        const candidatePaths = [
          ...discoveredEvaluationPaths,
          "explainability/evaluation/evaluation_summary.csv",
          "explainability/benchmark/benchmark_summary.csv",
          "explainability/benchmark/benchmark_results_summary.csv",
        ];

        for (const candidate of Array.from(new Set(candidatePaths))) {
          const csvUrl = artifactUrl(runId, candidate.replace(/\\/g, "/"));
          const resp = await fetch(csvUrl);
          if (!resp.ok) continue;
          const text = await resp.text();
          if (cancelled) return;
          const parsed = parseCsv(text);
          if (!parsed) continue;
          setEvaluationHeaders(parsed.headers);
          setEvaluationRows(parsed.rows);
          return;
        }
      } catch (e) {
        if (!cancelled) {
          setEvaluationError(e instanceof Error ? e.message : String(e));
        }
      }
    };
    loadEvaluation();
    return () => {
      cancelled = true;
    };
  }, [runId, artifacts]);

  useEffect(() => {
    if (!activeImage) return;
    setZoom(1);
    pinchState.current.active = false;
    pinchState.current.startDistance = 0;
    pinchState.current.startZoom = 1;
    pinchState.current.pointers.clear();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setActiveImage(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeImage]);

  const clampZoom = (value: number) => Math.min(4, Math.max(0.5, value));

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!activeImage) return;
    event.preventDefault();
    const delta = -event.deltaY;
    const next = clampZoom(zoom + (delta > 0 ? 0.1 : -0.1));
    setZoom(next);
  };

  const updatePinchZoom = () => {
    if (pinchState.current.pointers.size < 2) return;
    const points = Array.from(pinchState.current.pointers.values());
    const dx = points[0].x - points[1].x;
    const dy = points[0].y - points[1].y;
    const distance = Math.hypot(dx, dy);
    if (!pinchState.current.active) {
      pinchState.current.active = true;
      pinchState.current.startDistance = distance;
      pinchState.current.startZoom = zoom;
      return;
    }
    if (pinchState.current.startDistance <= 0) return;
    const scale = distance / pinchState.current.startDistance;
    setZoom(clampZoom(pinchState.current.startZoom * scale));
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!activeImage) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pinchState.current.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    updatePinchZoom();
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!activeImage) return;
    if (!pinchState.current.pointers.has(event.pointerId)) return;
    pinchState.current.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    updatePinchZoom();
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (!activeImage) return;
    pinchState.current.pointers.delete(event.pointerId);
    if (pinchState.current.pointers.size < 2) {
      pinchState.current.active = false;
      pinchState.current.startDistance = 0;
      pinchState.current.startZoom = zoom;
    }
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-brand-50">
      <div className="px-8 pt-6 shrink-0">
        <h2 className="text-2xl font-semibold">Results</h2>
        <p className="text-sm text-brand-600 mt-1">
          Your run artifacts are ready. A zip download starts automatically when the run finishes.
        </p>
      </div>

      <div className="flex-1 overflow-auto min-w-0">
        <div className="w-full px-8 py-6">
          <div className="border border-brand-100 rounded-xl p-6 bg-white space-y-4">
            <div className="text-sm text-gray-700">
              <span className="font-medium">Run ID:</span> {runId ?? "-"}
            </div>

            {runId && (
              <a
                className="inline-flex px-4 py-2 rounded-md text-sm border border-brand-300 bg-brand-50 text-brand-700 hover:bg-brand-100"
                href={artifactsZipUrl(runId)}
                download
              >
                Download results zip
              </a>
            )}

            <ResultsSummary
              summaryData={summaryData}
              metricsData={metricsData}
              datasetSummaryData={datasetSummaryData}
              trainingSummaryData={trainingSummaryData}
            />

            <div className="border-t pt-4 space-y-6">
              <div className="text-sm font-medium text-gray-800">Result Images</div>
              {loading && <div className="text-sm text-brand-600">Loading images...</div>}
              {error && (
                <div className="text-sm text-red-600">Failed to load images: {error}</div>
              )}
              {!loading && !error && pngs.length === 0 && (
                <div className="text-sm text-brand-600">No PNG images found for this run.</div>
              )}
              {!loading && !error && pngs.length > 0 && (
                <>
                  <ImageSection
                    title="Training"
                    paths={pngs.filter(
                      (p) =>
                        normalizePath(p).includes("training") ||
                        normalizePath(p).includes("history")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="SHAP Explainability Results"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/shap")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="LIME Explainability Results"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/lime")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="Transformer Temporal Attribution"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/temporal_attribution")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="GNN Gradient Explainability Results"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/gradient")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="GNN Temporal Explainability Results"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/temporal/")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="GNN GraphLIME Explainability Results"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/graphlime")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="GNN Local Graph Explanations"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/prophet/") &&
                      normalizePath(p).includes("/local/")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                  <ImageSection
                    title="GNN Global Explainability"
                    paths={pngs.filter((p) =>
                      normalizePath(p).includes("explainability/prophet/") &&
                      normalizePath(p).includes("/global/")
                    )}
                    runId={runId}
                    onOpen={setActiveImage}
                  />
                </>
              )}
              <EvaluationSection
                headers={evaluationHeaders}
                rows={evaluationRows}
                error={evaluationError}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="shrink-0 px-8 pb-6 border-t border-brand-100 bg-white">
        <div className="flex items-center justify-between gap-4 pt-4">
          <div className="flex flex-wrap items-center gap-3">
            <button
              className="px-4 py-2 border border-brand-200 rounded-md text-sm text-brand-700 hover:bg-brand-50"
              onClick={onBackToPipeline}
            >
              Back to Pipeline
            </button>
            <button
              className="px-4 py-2 border border-brand-300 rounded-md text-sm text-brand-700 bg-white hover:bg-brand-50"
              onClick={onRerun}
            >
              Rerun Pipeline
            </button>
            <button
              className="px-4 py-2 border border-gray-300 rounded-md text-sm text-gray-700 bg-white hover:bg-gray-50"
              onClick={onFinish}
            >
              Finish
            </button>
          </div>
          <p className="text-sm text-gray-500">
            Built with <span className="text-pink-500">❤</span> at TUM
          </p>
        </div>
      </div>

      {activeImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
          onClick={() => setActiveImage(null)}
          onWheel={handleWheel}
        >
          <button
            className="fixed right-4 top-4 z-10 inline-flex h-10 w-10 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 shadow hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-brand-500"
            onClick={() => setActiveImage(null)}
            aria-label="Close image preview"
            type="button"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
          <div
            className="relative max-h-full max-w-6xl w-full"
            onClick={(e) => e.stopPropagation()}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            style={{ touchAction: "none" }}
          >
            <button
              className="hidden"
              onClick={() => setActiveImage(null)}
              aria-label="Close image preview"
              type="button"
            >
              ×
            </button>
            <img
              src={activeImage.src}
              alt={activeImage.name}
              className="w-full h-auto max-h-[85vh] object-contain rounded-lg bg-white"
              style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
            />
            <div className="mt-3 text-xs text-gray-200 text-center break-all">
              {activeImage.name}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function normalizePath(path: string) {
  return path.replace(/\\/g, "/").toLowerCase();
}

function parseCsv(text: string) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return null;
  const headers = parseCsvLine(lines[0]).map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const cols = parseCsvLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = (cols[i] ?? "").trim();
    });
    return row;
  });
  return rows.length ? { headers, rows } : null;
}

function parseCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      cells.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  cells.push(current);
  return cells;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function formatNumber(value: unknown) {
  const num = asNumber(value);
  if (num === null) return "N/A";
  return num.toLocaleString();
}

function formatDecimal(value: unknown, digits = 4) {
  const num = asNumber(value);
  if (num === null) return "N/A";
  return num.toFixed(digits);
}

function formatPercent(value: unknown, digits = 1) {
  const num = asNumber(value);
  if (num === null) return "N/A";
  return `${(num * 100).toFixed(digits)}%`;
}

function formatImprovement(modelValue: unknown, baselineValue: unknown) {
  const model = asNumber(modelValue);
  const baseline = asNumber(baselineValue);
  if (model === null || baseline === null || baseline <= 0) return null;
  const improvement = ((baseline - model) / baseline) * 100;
  const direction = improvement >= 0 ? "better" : "worse";
  const magnitude = Math.abs(improvement);
  let verdict = "Weak";
  if (improvement >= 40) verdict = "Strong";
  else if (improvement >= 20) verdict = "Moderate";
  else if (improvement >= 0) verdict = "Small";
  else verdict = "Worse";
  return `${magnitude.toFixed(1)}% ${direction} (${verdict})`;
}

function titleFromTask(task: unknown) {
  const raw = asString(task);
  if (!raw) return "N/A";
  return raw.replace(/_/g, " ");
}

function ResultsSummary({
  summaryData,
  metricsData,
  datasetSummaryData,
  trainingSummaryData,
}: {
  summaryData: Record<string, unknown> | null;
  metricsData: Record<string, unknown> | null;
  datasetSummaryData: Record<string, unknown> | null;
  trainingSummaryData: Record<string, unknown> | null;
}) {
  const summary = asRecord(summaryData);
  const metricsRoot = asRecord(metricsData);
  const datasetSummary = asRecord(datasetSummaryData);
  const trainingSummary = asRecord(trainingSummaryData);
  const dataset = asRecord(summary?.dataset);
  const request = asRecord(summary?.request);
  const split = asRecord(metricsRoot?.split);
  const config = asRecord(metricsRoot?.config);
  const metrics = asRecord(metricsRoot?.metrics) ?? asRecord(summary?.metrics);
  const taskName = asString(request?.task ?? metricsRoot?.task ?? summary?.request);

  const numEvents = asNumber(dataset?.num_events);
  const numCases = asNumber(dataset?.num_cases);
  const avgEventsPerCase =
    numEvents !== null && numCases !== null && numCases > 0 ? numEvents / numCases : null;

  const fallbackDatasetRows = [
    { label: "Dataset", value: asString(dataset?.filename) ?? "N/A" },
    { label: "Cases", value: formatNumber(dataset?.num_cases) },
    { label: "Events", value: formatNumber(dataset?.num_events) },
    { label: "Avg. events per case", value: avgEventsPerCase === null ? "N/A" : avgEventsPerCase.toFixed(2) },
    { label: "Model type", value: asString(request?.model_type) ?? asString(metricsRoot?.model_type) ?? "N/A" },
    { label: "Task", value: titleFromTask(request?.task ?? metricsRoot?.task) },
    { label: "Explainability", value: asString(request?.explainability) ?? "N/A" },
    { label: "Finished", value: asString(summary?.finished_at) ?? asString(metricsRoot?.finished_at) ?? "N/A" },
  ];

  const fallbackTrainingRows = [
    { label: "Epochs", value: formatNumber(config?.epochs) },
    { label: "Batch size", value: formatNumber(config?.batch_size) },
    { label: "Patience", value: formatNumber(config?.patience) },
    { label: "Learning rate", value: formatDecimal(config?.lr, 5) },
    { label: "Dropout", value: formatDecimal(config?.dropout_rate, 3) },
    { label: "Hidden size", value: formatNumber(config?.hidden ?? config?.d_model) },
    { label: "Attention heads", value: formatNumber(config?.heads ?? config?.num_heads) },
    { label: "Layers/Blocks", value: formatNumber(config?.num_layers ?? config?.num_blocks) },
    { label: "Test split", value: formatPercent(split?.test_size) },
    { label: "Validation split", value: formatPercent(split?.val_split) },
  ];

  const datasetRows = extractSummaryRows(datasetSummary) ?? fallbackDatasetRows;
  const trainingRows = extractSummaryRows(trainingSummary) ?? fallbackTrainingRows;
  const visibleDatasetRows =
    taskName === "next_activity" || taskName === "custom_activity"
      ? datasetRows.filter((row) => row.label !== "Proportion of Positive Cases")
      : datasetRows;
  const visibleTrainingRows = filterRowsForSelectedTask(trainingRows, taskName);

  const metricRows = buildMetricRows(metrics, taskName);

  if (!summary && !metricsRoot) return null;

  return (
    <section className="space-y-6 border-t pt-4">
      <div className="grid gap-6 xl:grid-cols-2">
        <SummaryTable title="Dataset Summary" rows={visibleDatasetRows} showAllRows />
        <SummaryTable title="Training Summary" rows={visibleTrainingRows} showAllRows />
      </div>
      {metricRows.length > 0 ? (
        <SummaryTable title="Performance Summary" rows={metricRows} compact />
      ) : null}
    </section>
  );
}

function extractSummaryRows(data: Record<string, unknown> | null) {
  const rows = data?.rows;
  if (!Array.isArray(rows)) return null;
  const normalized = rows
    .map((row) => {
      const record = asRecord(row);
      if (!record) return null;
      return {
        label: asString(record.label) ?? "N/A",
        value:
          typeof record.value === "string" || typeof record.value === "number"
            ? String(record.value)
            : "N/A",
      };
    })
    .filter((row): row is { label: string; value: string } => row !== null);
  return normalized.length ? normalized : null;
}

function filterRowsForSelectedTask(
  rows: Array<{ label: string; value: string }>,
  taskName?: string | null
) {
  if (taskName === "unified") return rows;

  const isEventTime = taskName === "event_time";
  const isRemainingTime = taskName === "remaining_time";
  const isActivity = taskName === "next_activity" || taskName === "custom_activity";

  return rows.filter((row) => {
    const label = row.label.toLowerCase();
    const isEventMetric = label.includes("event time");
    const isRemainingMetric = label.includes("remaining time");
    const isAccuracyMetric = label.includes("accuracy");

    if (isEventTime) {
      if (isRemainingMetric || isAccuracyMetric) return false;
      return true;
    }
    if (isRemainingTime) {
      if (isEventMetric || isAccuracyMetric) return false;
      return true;
    }
    if (isActivity) {
      if (isEventMetric || isRemainingMetric) return false;
      return true;
    }
    return true;
  });
}

function buildMetricRows(metrics: Record<string, unknown> | null, taskName?: string | null) {
  if (!metrics) return [];
  const rows: Array<{ label: string; value: string }> = [];
  const isNextActivityTask =
    taskName === "next_activity" || taskName === "custom_activity";
  const isEventTimeTask = taskName === "event_time";
  const isRemainingTimeTask = taskName === "remaining_time";
  const isUnifiedTask = taskName === "unified";
  const accuracy = metrics.accuracy;
  const loss = metrics.loss ?? metrics.test_loss;
  const maeTime = metrics.mae_time ?? (isEventTimeTask ? metrics.test_mae : undefined);
  const maeRem = metrics.mae_rem ?? (isRemainingTimeTask ? metrics.test_mae : undefined);
  const eventMeanBaseline = metrics.event_time_baseline_mean_mae;
  const eventMedianBaseline = metrics.event_time_baseline_median_mae;
  const remMeanBaseline = metrics.remaining_time_baseline_mean_mae;
  const remMedianBaseline = metrics.remaining_time_baseline_median_mae;
  const showAccuracy =
    taskName === "next_activity" ||
    taskName === "custom_activity" ||
    isUnifiedTask ||
    (!taskName && accuracy !== undefined);
  if (showAccuracy && accuracy !== undefined) {
    rows.push({ label: "Accuracy", value: formatPercent(accuracy, 2) });
  }
  const showEventTimeMetrics = isUnifiedTask || isEventTimeTask || (!taskName && maeTime !== undefined);
  const showRemainingTimeMetrics = isUnifiedTask || isRemainingTimeTask || (!taskName && maeRem !== undefined);

  if (showEventTimeMetrics && maeTime !== undefined) {
    rows.push({ label: "Event time MAE", value: formatDecimal(maeTime) });
  }
  if (showEventTimeMetrics && eventMeanBaseline !== undefined) {
    rows.push({ label: "Event time mean baseline MAE", value: formatDecimal(eventMeanBaseline) });
    const summary = formatImprovement(maeTime, eventMeanBaseline);
    if (summary) {
      rows.push({ label: "Event time vs mean baseline", value: summary });
    }
  }
  if (showEventTimeMetrics && eventMedianBaseline !== undefined) {
    rows.push({ label: "Event time median baseline MAE", value: formatDecimal(eventMedianBaseline) });
    const summary = formatImprovement(maeTime, eventMedianBaseline);
    if (summary) {
      rows.push({ label: "Event time vs median baseline", value: summary });
    }
  }
  if (showRemainingTimeMetrics && maeRem !== undefined) {
    rows.push({ label: "Remaining time MAE", value: formatDecimal(maeRem) });
  }
  if (showRemainingTimeMetrics && remMeanBaseline !== undefined) {
    rows.push({ label: "Remaining time mean baseline MAE", value: formatDecimal(remMeanBaseline) });
    const summary = formatImprovement(maeRem, remMeanBaseline);
    if (summary) {
      rows.push({ label: "Remaining time vs mean baseline", value: summary });
    }
  }
  if (showRemainingTimeMetrics && remMedianBaseline !== undefined) {
    rows.push({ label: "Remaining time median baseline MAE", value: formatDecimal(remMedianBaseline) });
    const summary = formatImprovement(maeRem, remMedianBaseline);
    if (summary) {
      rows.push({ label: "Remaining time vs median baseline", value: summary });
    }
  }
  if (loss !== undefined) {
    rows.push({ label: "Loss", value: formatDecimal(loss) });
  }
  for (const [key, value] of Object.entries(metrics)) {
    if (
      [
        "accuracy",
        "loss",
        "test_loss",
        "mae_time",
        "mae_rem",
        "test_mae",
        "event_time_baseline_mean_value",
        "event_time_baseline_median_value",
        "event_time_baseline_mean_mae",
        "event_time_baseline_median_mae",
        "remaining_time_baseline_mean_value",
        "remaining_time_baseline_median_value",
        "remaining_time_baseline_mean_mae",
        "remaining_time_baseline_median_mae",
      ].includes(key)
    ) {
      continue;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      const lowerKey = key.toLowerCase();
      if (isNextActivityTask && (lowerKey.includes("mae") || lowerKey.includes("event_time") || lowerKey.includes("remaining_time"))) {
        continue;
      }
      if (isEventTimeTask && (lowerKey.includes("remaining_time") || lowerKey.includes("mae_rem") || lowerKey.includes("accuracy"))) {
        continue;
      }
      if (isRemainingTimeTask && (lowerKey.includes("event_time") || lowerKey.includes("mae_time") || lowerKey.includes("accuracy"))) {
        continue;
      }
      rows.push({ label: key.replace(/_/g, " "), value: formatDecimal(value) });
    }
  }
  return rows;
}

function SummaryTable({
  title,
  rows,
  compact = false,
  showAllRows = false,
}: {
  title: string;
  rows: Array<{ label: string; value: string }>;
  compact?: boolean;
  showAllRows?: boolean;
}) {
  const filtered = showAllRows ? rows : rows.filter((row) => row.value !== "N/A");
  if (!filtered.length) return null;
  return (
    <section className="space-y-3">
      <div className="text-sm font-semibold text-brand-900">{title}</div>
      <div className="overflow-auto border border-brand-100 rounded-lg bg-white">
        <table className="min-w-full text-sm">
          <tbody>
            {filtered.map((row) => (
              <tr key={`${title}-${row.label}`} className="odd:bg-white even:bg-brand-50/40">
                <td className={`px-3 py-2 border-b border-brand-100 text-gray-700 font-medium ${compact ? "w-1/3" : "w-1/2"}`}>
                  {row.label}
                </td>
                <td className="px-3 py-2 border-b border-brand-100 text-gray-700">{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ImageSection({
  title,
  paths,
  runId,
  onOpen,
}: {
  title: string;
  paths: string[];
  runId: string | null;
  onOpen: (img: { src: string; name: string } | null) => void;
}) {
  if (!paths.length) return null;

  return (
    <section className="space-y-3">
      <div className="text-sm font-semibold text-brand-900">{title}</div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {paths.map((path) => {
          const filename = normalizePath(path).split("/").pop() ?? path;
          return (
            <div key={path} className="border border-brand-100 rounded-lg p-3 bg-brand-50">
              <div className="text-xs text-gray-600 mb-2">{filename}</div>
              <img
                src={artifactUrl(runId ?? "", path)}
                alt={filename}
                className="w-full h-auto rounded-md border border-brand-100 bg-white cursor-zoom-in"
                loading="lazy"
                onClick={() =>
                  onOpen({
                    src: artifactUrl(runId ?? "", path),
                    name: filename,
                  })
                }
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function EvaluationSection({
  headers,
  rows,
  error,
}: {
  headers: string[];
  rows: Array<Record<string, string>>;
  error: string | null;
}) {
  if (error) {
    return (
      <section className="space-y-2">
        <div className="text-sm font-semibold text-brand-900">Evaluation</div>
        <div className="text-sm text-red-600">Failed to load evaluation: {error}</div>
      </section>
    );
  }
  if (!headers.length || !rows.length) return null;

  const evaluationTable = buildEvaluationMatrix(rows);
  if (!evaluationTable.metricRows.length || !evaluationTable.kColumns.length) return null;

  return (
    <section className="space-y-3">
      <div className="text-sm font-semibold text-brand-900">Evaluation</div>
      <div className="text-xs text-brand-600">
        Columns are k-values. Each cell shows the metric value and n=valid_sample_count.
      </div>
      {evaluationTable.metricRows.length ? (
        <div className="overflow-auto border border-brand-100 rounded-lg bg-white">
          <table className="min-w-full text-sm">
            <thead className="bg-brand-50 text-brand-800">
              <tr>
                <th className="text-left px-3 py-2 font-medium border-b border-brand-100">Metric</th>
                {evaluationTable.kColumns.map((k) => (
                  <th
                    key={k}
                    className="text-left px-3 py-2 font-medium border-b border-brand-100 min-w-[170px]"
                  >
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {evaluationTable.metricRows.map((row) => (
                <tr key={row.metricKey} className="odd:bg-white even:bg-brand-50/40">
                  <td
                    className="px-3 py-2 border-b border-brand-100 text-gray-800 font-medium"
                    title={metricHelp(row.metricKey)}
                  >
                    {row.label}
                  </td>
                  {evaluationTable.kColumns.map((k) => {
                    const cell = row.values[k];
                    return (
                      <td key={`${row.metricKey}-${k}`} className="px-3 py-2 border-b border-brand-100 text-gray-700 align-top">
                        <div>{cell?.value ?? "-"}</div>
                        <div className="text-xs text-gray-500">
                          {cell?.validSampleCount ? `n=${cell.validSampleCount}` : "n=-"}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function buildEvaluationMatrix(rows: Array<Record<string, string>>) {
  const kColumns = Array.from(
    new Set(
      rows
        .map((row) => String(row.metric || row.Metric || row.METRIC || "").toLowerCase().match(/(?:^|_)(k\d+)(?:_|$)/)?.[1])
        .filter((k): k is string => Boolean(k))
    )
  ).sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  const metricRows = [
    { metricKey: "spearman_correlation", label: "Faithfulness Spearman" },
    { metricKey: "pearson_correlation", label: "Faithfulness Pearson" },
    { metricKey: "comprehensiveness_mean", label: "Comprehensiveness" },
    { metricKey: "sufficiency_mean", label: "Sufficiency" },
    { metricKey: "jaccard_similarity", label: "Agreement Jaccard" },
    { metricKey: "top_k_overlap", label: "Agreement Overlap" },
  ];

  const structuredMetricRows = metricRows.map((metric) => ({
    metricKey: metric.metricKey,
    label: metric.label,
    values: Object.fromEntries(kColumns.map((k) => [k, null])) as Record<
      string,
      { value: string; validSampleCount: string | null } | null
    >,
  }));

  for (const row of rows) {
    const metric = ((row.metric || row.Metric || row.METRIC || "") as string).toLowerCase();
    const rawValue = row.value || row.Value || row.VALUE || "";
    const num = Number(rawValue);
    const kMatch = metric.match(/(?:^|_)(k\d+)(?:_|$)/);
    if (!kMatch) continue;
    const kKey = kMatch[1];
    if (!kColumns.includes(kKey)) continue;

    const validCountTargets =
      metric.includes("valid_sample_count") && metric.includes("faithfulness")
        ? ["spearman_correlation", "pearson_correlation"]
        : metric.includes("valid_sample_count") && metric.includes("comprehensiveness")
          ? ["comprehensiveness_mean"]
          : metric.includes("valid_sample_count") && metric.includes("sufficiency")
            ? ["sufficiency_mean"]
            : metric.includes("valid_sample_count") && metric.includes("agreement")
              ? ["jaccard_similarity", "top_k_overlap"]
              : [];

    if (validCountTargets.length) {
      for (const targetKey of validCountTargets) {
        const targetMetricRow = structuredMetricRows.find((entry) => entry.metricKey === targetKey);
        if (!targetMetricRow) continue;
        const currentCell = targetMetricRow.values[kKey] ?? { value: "-", validSampleCount: null };
        targetMetricRow.values[kKey] = {
          ...currentCell,
          validSampleCount: Number.isFinite(num) ? String(Math.trunc(num)) : "-",
        };
      }
      continue;
    }

    const resolvedColumn =
      metric.includes("spearman_correlation")
        ? "spearman_correlation"
        : metric.includes("pearson_correlation")
          ? "pearson_correlation"
          : metric.includes("comprehensiveness") && metric.endsWith("_mean")
            ? "comprehensiveness_mean"
            : metric.includes("sufficiency") && metric.endsWith("_mean")
              ? "sufficiency_mean"
              : metric.includes("jaccard_similarity")
                ? "jaccard_similarity"
                : metric.includes("top_k_overlap")
                  ? "top_k_overlap"
                  : null;

    if (!resolvedColumn) continue;
    const targetMetricRow = structuredMetricRows.find((entry) => entry.metricKey === resolvedColumn);
    if (!targetMetricRow) continue;
    const currentCell = targetMetricRow.values[kKey] ?? { value: "-", validSampleCount: null };
    targetMetricRow.values[kKey] = {
      ...currentCell,
      value: Number.isFinite(num) ? num.toFixed(4) : "-",
    };
  }

  return { kColumns, metricRows: structuredMetricRows };
}

function metricHelp(metric: string) {
  const key = metric.toLowerCase();
  if (key.includes("sparsity_score")) {
    return "Sparsity: whether the explanation concentrates importance on a small subset of nodes. Higher means a more compact explanation; 0 means the explanation is dense.";
  }
  if (key.includes("spearman") || key.includes("pearson")) {
    return "Faithfulness: correlation between importance scores and model output change. Higher is better.";
  }
  if (key.includes("jaccard") || key.includes("overlap")) {
    return "Agreement: overlap of top-k features between explainers. Higher is better.";
  }
  if (key.includes("temporal_consistency")) {
    return "Temporal recency correlation: positive values mean later/recent events are more important; negative values mean earlier events are more important. This is descriptive, not always a good/bad score.";
  }
  return "Evaluation metric.";
}

