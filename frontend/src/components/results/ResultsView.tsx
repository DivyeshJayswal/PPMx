import { useEffect, useRef, useState, type PointerEvent, type WheelEvent } from "react";

import { artifactUrl, artifactsZipUrl, listArtifacts } from "../../lib/api";

type ResultsViewProps = {
  runId: string | null;
  onBackToPipeline: () => void;
};

export default function ResultsView({ runId, onBackToPipeline }: ResultsViewProps) {
  const [pngs, setPngs] = useState<string[]>([]);
  const [benchmarkRows, setBenchmarkRows] = useState<Array<Record<string, string>>>([]);
  const [benchmarkHeaders, setBenchmarkHeaders] = useState<string[]>([]);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
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
        setPngs([]);
        setBenchmarkRows([]);
        setBenchmarkHeaders([]);
        return;
      }
      setLoading(true);
      setError(null);
      setBenchmarkError(null);
      try {
        const res = await listArtifacts(runId);
        if (cancelled) return;
        const images = (res.artifacts ?? []).filter((path) =>
          path.toLowerCase().endsWith(".png")
        );
        setPngs(images);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setPngs([]);
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
    const loadBenchmark = async () => {
      if (!runId) return;
      setBenchmarkError(null);
      try {
        const csvUrl = artifactUrl(runId, "explainability/benchmark/benchmark_summary.csv");
        const resp = await fetch(csvUrl);
        if (!resp.ok) return;
        const text = await resp.text();
        if (cancelled) return;
        const lines = text.trim().split(/\r?\n/).filter(Boolean);
        if (!lines.length) return;
        const headers = lines[0].split(",").map((h) => h.trim());
        const rows = lines.slice(1).map((line) => {
          const cols = line.split(",");
          const row: Record<string, string> = {};
          headers.forEach((h, i) => {
            row[h] = (cols[i] ?? "").trim();
          });
          return row;
        });
        setBenchmarkHeaders(headers);
        setBenchmarkRows(rows);
      } catch (e) {
        if (!cancelled) {
          setBenchmarkError(e instanceof Error ? e.message : String(e));
        }
      }
    };
    loadBenchmark();
    return () => {
      cancelled = true;
    };
  }, [runId]);

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
                  <BenchmarkSection
                    headers={benchmarkHeaders}
                    rows={benchmarkRows}
                    error={benchmarkError}
                  />
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="shrink-0 px-8 pb-6 border-t border-brand-100 bg-white">
        <div className="flex items-center justify-between gap-4 pt-4">
          <button
            className="px-4 py-2 border border-brand-200 rounded-md text-sm text-brand-700 hover:bg-brand-50"
            onClick={onBackToPipeline}
          >
            Back to Pipeline
          </button>
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
              className="absolute -top-3 -right-3 h-9 w-9 rounded-full bg-white text-gray-700 border shadow hover:bg-gray-50"
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

function BenchmarkSection({
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
        <div className="text-sm font-semibold text-brand-900">Benchmark</div>
        <div className="text-sm text-red-600">Failed to load benchmark: {error}</div>
      </section>
    );
  }
  if (!headers.length || !rows.length) return null;

  const compactRows = buildCompactBenchmark(rows);
  if (!compactRows.length) return null;

  return (
    <section className="space-y-3">
      <div className="text-sm font-semibold text-brand-900">Benchmark</div>
      <div className="text-xs text-brand-600">Showing k10 metrics only.</div>
      <div className="overflow-auto border border-brand-100 rounded-lg bg-white">
        <table className="min-w-full text-sm">
          <thead className="bg-brand-50 text-brand-800">
            <tr>
              <th className="text-left px-3 py-2 font-medium border-b border-brand-100">Metric</th>
              <th className="text-left px-3 py-2 font-medium border-b border-brand-100">Value</th>
              <th className="text-left px-3 py-2 font-medium border-b border-brand-100">
                Best
              </th>
              <th className="text-left px-3 py-2 font-medium border-b border-brand-100">
                Feedback
              </th>
            </tr>
          </thead>
          <tbody>
            {compactRows.map((row, idx) => (
              <tr key={idx} className="odd:bg-white even:bg-brand-50/40">
                <td
                  className="px-3 py-2 border-b border-brand-100 text-gray-700"
                  title={metricHelp(row.metric)}
                >
                  {row.metric}
                </td>
                <td className="px-3 py-2 border-b border-brand-100 text-gray-700">{row.value}</td>
                <td className="px-3 py-2 border-b border-brand-100 text-gray-700">
                  {row.best}
                </td>
                <td className="px-3 py-2 border-b border-brand-100 text-gray-700">
                  {row.feedback}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function buildCompactBenchmark(rows: Array<Record<string, string>>) {
  const compact = rows.filter((row) => {
    const metric = row.metric || row.Metric || row.METRIC || "";
    const normalized = metric.toLowerCase();
    if (!normalized.includes("k10") && !normalized.includes("temporal_consistency")) {
      return false;
    }
    if (
      normalized.includes("p_value") ||
      normalized.includes("mean") ||
      normalized.includes("median") ||
      normalized.includes("std") ||
      normalized.includes("position") ||
      normalized.includes("rank_correlation")
    ) {
      return false;
    }
    return true;
  });

  return compact.map((row) => {
    const metric = row.metric || row.Metric || row.METRIC || "";
    const rawValue = row.value || row.Value || row.VALUE || "";
    const num = Number(rawValue);
    const value = Number.isFinite(num) ? num.toFixed(4) : rawValue;
    const feedback = Number.isFinite(num) ? scoreFeedback(metric, num) : "N/A";
    const best = bestTarget(metric);
    return { metric, value, best, feedback };
  });
}

function scoreFeedback(metric: string, value: number) {
  const key = metric.toLowerCase();
  if (key.includes("temporal_consistency")) {
    if (value >= 0.7) return "Good";
    if (value >= 0.4) return "Fair";
    return "Weak";
  }
  if (key.includes("jaccard") || key.includes("overlap")) {
    if (value >= 0.6) return "Good";
    if (value >= 0.4) return "Fair";
    if (value >= 0.2) return "Weak";
    return "Poor";
  }
  if (key.includes("correlation")) {
    if (value >= 0.5) return "Good";
    if (value >= 0.2) return "Fair";
    if (value >= 0) return "Weak";
    return "Poor";
  }
  return "N/A";
}

function bestTarget(metric: string) {
  const key = metric.toLowerCase();
  if (key.includes("jaccard") || key.includes("overlap")) return "1.0";
  if (key.includes("correlation")) return "1.0";
  if (key.includes("temporal_consistency")) return "1.0";
  return "N/A";
}

function metricHelp(metric: string) {
  const key = metric.toLowerCase();
  if (key.includes("spearman") || key.includes("pearson")) {
    return "Faithfulness: correlation between importance scores and model output change. Higher is better.";
  }
  if (key.includes("jaccard") || key.includes("overlap")) {
    return "Agreement: overlap of top-k features between explainers. Higher is better.";
  }
  if (key.includes("temporal_consistency")) {
    return "Temporal consistency: whether recent events are consistently more important. Higher is better.";
  }
  return "Benchmark metric.";
}

