// frontend/src/components/pages/wizardLayout.tsx
import { useEffect, useMemo, useState } from "react";

import Sidebar from "../layout/Sidebar";

import Step1Upload from "../steps/Step1Upload";
import { type ManualMapping, type MappingMode } from "../steps/Step2Mapping";
import Step2Model from "../steps/Step2Model";
import Step3Prediction from "../steps/Step3Prediction";
import Step4Explainability, { type ExplainValue } from "../steps/Step4Explainability";
import Step5Config, {
  type ConfigMode,
  type GnnConfig,
  type TransformerConfig,
} from "../steps/Step5Config";
import Step5ExplainabilityConfig, {
  type ExplainabilityConfig,
} from "../steps/Step5ExplainabilityConfig";
import Step6Review from "../steps/Step6Review";
import ResultsView from "../results/ResultsView";

import StepProgressHeader from "../ui/StepProgressHeader";
import WizardFooter from "../ui/WizardFooter";
import ppmxLogo from "../../assets/ppmx.png";
import tumLogo from "../../assets/tum.png";

import {
  artifactsZipUrl,
  createRun,
  getRun,
  getRunLogs,
  listArtifacts,
  type DatasetUploadResponse,
  type RunStatus,
} from "../../lib/api";

const TOTAL_STEPS = 8;

export type PipelineStatus = "idle" | "running" | "completed";
export type ViewMode = "wizard" | "results";

const ANSI_RE = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, "g");

function cleanLogLine(line: string): string {
  return line.replace(ANSI_RE, "").replace(/\r/g, "").trim();
}

function isNoiseLine(line: string): boolean {
  if (!line) return true;
  if (line.includes("ms/step")) return true;
  if (line.includes("====")) return true;
  if (line.includes("ETA") || line.includes("eta")) return true;
  return false;
}

function extractEpochProgress(lines: string[]): { current: number; total: number } | null {
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = lines[i].match(/Epoch\s+(\d+)\s*\/\s*(\d+)/i);
    if (m) {
      const current = Number(m[1]);
      const total = Number(m[2]);
      if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
        return { current, total };
      }
    }
  }
  return null;
}

function extractGraphBuildProgress(lines: string[]): number | null {
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = lines[i].match(/Building graphs:\s+(\d+)%/i);
    if (m) {
      const pct = Number(m[1]);
      if (Number.isFinite(pct)) return Math.max(0, Math.min(100, pct));
    }
  }
  return null;
}

function detectExplainabilityProgress(lines: string[]): number | null {
  const joined = lines.join("\n").toLowerCase();

  if (
    joined.includes("combined evaluation summary saved") ||
    joined.includes("evaluation summary saved") ||
    joined.includes("evaluation results saved")
  ) {
    return 98;
  }
  if (joined.includes("[evaluation]") || joined.includes("running evaluation")) {
    return 93;
  }
  if (
    joined.includes("generating comprehensive analysis") ||
    joined.includes("generating comparison report") ||
    joined.includes("feature importance summary saved") ||
    joined.includes("global view importance")
  ) {
    return 90;
  }
  if (
    joined.includes("graphlime local analysis") ||
    joined.includes("local explanation") ||
    joined.includes("sample_0_explanation") ||
    joined.includes("running lime") ||
    joined.includes("generating lime explanations")
  ) {
    return 87;
  }
  if (
    joined.includes("temporal gradient attribution") ||
    joined.includes("temporal gradient plots saved") ||
    joined.includes("shap_temporal_evolution")
  ) {
    return 83;
  }
  if (
    joined.includes("gradient analysis") ||
    joined.includes("heterogeneous gnn") ||
    joined.includes("gnnexplainer") ||
    joined.includes("global view importance") ||
    joined.includes("running shap") ||
    joined.includes("computing shap values") ||
    joined.includes("gnn explainability") ||
    joined.includes("explainability module:")
  ) {
    return 76;
  }
  if (joined.includes("explainability")) {
    return 74;
  }

  return null;
}

function estimateProgressFromLogs(lines: string[], status: RunStatus | null): number {
  if (!status) return 0;
  if (status.status === "queued") return 8;
  if (status.status === "failed") return 100;
  if (status.status === "succeeded") return 100;

  const graphBuild = extractGraphBuildProgress(lines);
  const epoch = extractEpochProgress(lines);
  let progress = 15;

  if (graphBuild !== null) {
    progress = Math.max(progress, Math.round(8 + graphBuild * 0.12));
  }

  if (epoch) {
    const frac = Math.min(1, epoch.current / epoch.total);
    progress = Math.max(progress, 25 + frac * 35); // 25-60
  }

  const joined = lines.join("\n").toLowerCase();
  if (joined.includes("[ok] built")) {
    progress = Math.max(progress, 22);
  }
  if (joined.includes("training gnn") || joined.includes("building model") || joined.includes("training transformer")) {
    progress = Math.max(progress, 24);
  }
  if (joined.includes("training completed")) {
    progress = Math.max(progress, 62);
  }
  if (joined.includes("evaluating on test") || joined.includes("evaluating")) {
    progress = Math.max(progress, 66);
  }
  if (
    joined.includes("model saved to:") ||
    joined.includes("training history plot saved") ||
    joined.includes("results saved to:")
  ) {
    progress = Math.max(progress, 71);
  }
  const explainabilityProgress = detectExplainabilityProgress(lines);
  if (explainabilityProgress !== null) {
    progress = Math.max(progress, explainabilityProgress);
  }

  return Math.min(99, Math.round(progress));
}

function normalizeModelType(v: string | null): "gnn" | "transformer" | null {
  if (!v) return null;
  const s = v.toLowerCase().trim();
  if (s === "gnn" || s.includes("gnn")) return "gnn";
  if (s === "transformer" || s.includes("transformer")) return "transformer";
  return null;
}

function normalizeTask(
  v: string | null
): "next_activity" | "custom_activity" | "event_time" | "remaining_time" | "unified" | null {
  if (!v) return null;
  const s = v.toLowerCase().trim();

  if (s === "next_activity" || s.includes("next activity")) return "next_activity";
  if (s === "custom_activity" || s.includes("custom")) return "custom_activity";
  if (s === "event_time" || s.includes("event time") || s === "timestamp") return "event_time";
  if (s === "remaining_time" || s.includes("remaining time")) return "remaining_time";
  if (s === "unified") return "unified";

  return null;
}

function isExplainAllowed(
  explain: ExplainValue | null,
  model: "gnn" | "transformer" | null
): boolean {
  if (!explain) return true;
  if (!model) return true;

  if (model === "transformer") {
    return explain === "none" || explain === "lime" || explain === "shap" || explain === "all";
  }
  return explain === "none" || explain === "gradient" || explain === "lime" || explain === "all";
}

function validateManualMapping(m: ManualMapping): boolean {
  const requiredOk =
    m.case_id.trim().length > 0 && m.activity.trim().length > 0 && m.timestamp.trim().length > 0;
  if (!requiredOk) return false;

  const selected = [m.case_id, m.activity, m.timestamp, m.resource].filter(
    (v): v is string => !!v && v.trim().length > 0
  );
  return new Set(selected).size === selected.length;
}

function mappingFromDetected(resp: DatasetUploadResponse): ManualMapping {
  const dm = resp.detected_mapping ?? {};
  return {
    case_id: dm.case_id ?? "",
    activity: dm.activity ?? "",
    timestamp: dm.timestamp ?? "",
    resource: dm.resource ?? null,
  };
}

function validateTransformerConfig(cfg: TransformerConfig): boolean {
  const positiveInts = [
    cfg.max_len,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_blocks,
    cfg.epochs,
    cfg.batch_size,
    cfg.patience,
  ].every((v) => Number.isInteger(v) && v > 0);

  const dropoutOk =
    typeof cfg.dropout_rate === "number" && cfg.dropout_rate > 0 && cfg.dropout_rate < 1;

  return positiveInts && dropoutOk;
}

function validateGnnConfig(cfg: GnnConfig): boolean {
  const positiveInts = [cfg.hidden, cfg.heads, cfg.num_layers, cfg.epochs, cfg.batch_size, cfg.patience].every(
    (v) => Number.isInteger(v) && v > 0
  );

  const dropoutOk =
    typeof cfg.dropout_rate === "number" && cfg.dropout_rate > 0 && cfg.dropout_rate < 1;

  const lrOk = typeof cfg.lr === "number" && cfg.lr > 0;

  return positiveInts && dropoutOk && lrOk;
}

function validateExplainabilityConfig(
  cfg: ExplainabilityConfig,
  model: "gnn" | "transformer" | null,
  method: ExplainValue | null
): boolean {
  if (!method) return false;
  if (method === "none") return true;
  const strategyOk = ["evenly_spaced", "random", "manual", "diverse"].includes(
    cfg.evaluation_sampling_strategy
  );
  const seedOk = Number.isInteger(cfg.evaluation_random_seed);
  const protocolOk = cfg.evaluation_protocol_name.trim().length > 0;
  const manualOk =
    cfg.evaluation_sampling_strategy !== "manual" ||
    (cfg.evaluation_sample_indices.trim().length > 0 &&
      cfg.evaluation_sample_indices
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean)
        .every((part) => /^\d+$/.test(part)));

  if (model === "transformer") {
    const sampleOk =
      Number.isInteger(cfg.transformer_explanation_samples) &&
      cfg.transformer_explanation_samples >= 1;
    return sampleOk && strategyOk && seedOk && protocolOk && manualOk;
  }
  if (model !== "gnn") return true;

  const localOk = Number.isInteger(cfg.local_explanation_samples) && cfg.local_explanation_samples >= 0;
  const globalOk =
    Number.isInteger(cfg.global_explanation_sample_percent) &&
    cfg.global_explanation_sample_percent >= 1 &&
    cfg.global_explanation_sample_percent <= 100;
  const evaluationOk = Number.isInteger(cfg.evaluation_samples) && cfg.evaluation_samples >= 1;
  const minOk = Number.isInteger(cfg.min_prefix_length) && cfg.min_prefix_length >= 1;
  const maxOk =
    cfg.max_prefix_length === null ||
    (Number.isInteger(cfg.max_prefix_length) && cfg.max_prefix_length >= cfg.min_prefix_length);

  return localOk && globalOk && evaluationOk && minOk && maxOk && strategyOk && seedOk && protocolOk && manualOk;
}

export default function WizardLayout() {
  const [step, setStep] = useState(0);

  /* -------------------- STEP DATA -------------------- */
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);
  const [datasetMode, setDatasetMode] = useState<"raw" | "preprocessed" | "skip" | null>(null);
  const [splitConfig, setSplitConfig] = useState({ test_size: 0.1, val_split: 0.11 });

  const [mappingMode, setMappingMode] = useState<MappingMode | null>("manual");
  const [manualMapping, setManualMapping] = useState<ManualMapping>({
    case_id: "",
    activity: "",
    timestamp: "",
    resource: null,
  });

  const [modelType, setModelType] = useState<string | null>(null);
  const [predictionTask, setPredictionTask] = useState<string | null>(null);
  const [predictionCategory, setPredictionCategory] = useState<"classification" | "regression" | null>(null);
  const [customTargetColumn, setCustomTargetColumn] = useState<string | null>(null);
  const [explainMethod, setExplainMethod] = useState<ExplainValue | null>(null);
  const [configMode, setConfigMode] = useState<ConfigMode | null>(null);

  /* -------------------- CONFIG STATE -------------------- */
  const defaultTransformerConfig = useMemo<TransformerConfig>(
    () => ({
      max_len: 16,
      d_model: 64,
      num_heads: 4,
      num_blocks: 2,
      dropout_rate: 0.1,
      epochs: 5,
      batch_size: 128,
      patience: 10,
    }),
    []
  );

  const defaultGnnConfig = useMemo<GnnConfig>(
    () => ({
      hidden: 64,
      heads: 4,
      num_layers: 2,
      dropout_rate: 0.1,
      lr: 4e-4,
      epochs: 5,
      batch_size: 64,
      patience: 10,
    }),
    []
  );

  const defaultExplainabilityConfig = useMemo<ExplainabilityConfig>(
    () => ({
      local_explanation_samples: 5,
      global_explanation_sample_percent: 1,
      evaluation_samples: 10,
      min_prefix_length: 1,
      max_prefix_length: null,
      transformer_explanation_samples: 50,
      evaluation_sampling_strategy: "evenly_spaced",
      evaluation_random_seed: 42,
      evaluation_sample_indices: "",
      evaluation_protocol_name: "Perturbation-Based Explainability Evaluation",
    }),
    []
  );

  const [transformerConfig, setTransformerConfig] =
    useState<TransformerConfig>(defaultTransformerConfig);
  const [gnnConfig, setGnnConfig] = useState<GnnConfig>(defaultGnnConfig);
  const [explainabilityConfig, setExplainabilityConfig] =
    useState<ExplainabilityConfig>(defaultExplainabilityConfig);

  /* -------------------- RUN STATE -------------------- */
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>("idle");
  const [progress, setProgress] = useState(0);

  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [runLogs, setRunLogs] = useState<string[]>([]);
  const [autoDownloadedRunId, setAutoDownloadedRunId] = useState<string | null>(null);

  const [viewMode, setViewMode] = useState<ViewMode>("wizard");

  /* -------------------- DERIVED -------------------- */
  const modelTypeNormalized = useMemo(
    () => normalizeModelType(modelType),
    [modelType]
  );

  const taskNormalized = useMemo(
    () => normalizeTask(predictionTask),
    [predictionTask]
  );

  /* -------------------- NAVIGATION -------------------- */
  const nextStep = () => setStep((prev) => Math.min(prev + 1, TOTAL_STEPS - 1));
  const prevStep = () => setStep((prev) => Math.max(prev - 1, 0));

  /* -------------------- VALIDATION -------------------- */
  const isStepValid = (stepIndex = step) => {
    switch (stepIndex) {
      case 0:
        return dataset !== null && !!dataset.split_paths && validateManualMapping(manualMapping);
      case 1:
        return modelTypeNormalized !== null;
      case 2: {
        if (!modelTypeNormalized) return false;
        if (configMode === null) return false;

        if (modelTypeNormalized === "transformer") {
          return configMode === "default" || validateTransformerConfig(transformerConfig);
        }
        if (modelTypeNormalized === "gnn") {
          return configMode === "default" || validateGnnConfig(gnnConfig);
        }
        return false;
      }
      case 3:
        if (!taskNormalized) return false;
        if (taskNormalized === "custom_activity") {
          if (!customTargetColumn) return false;
          if (!dataset?.column_types) return false;
          if (dataset.column_types[customTargetColumn] !== "categorical") return false;
        }
        return true;
      case 4:
        return explainMethod !== null;
      case 5:
        return validateExplainabilityConfig(explainabilityConfig, modelTypeNormalized, explainMethod);
      case 6:
        return pipelineStatus === "completed";
      case 7:
        return true;
      default:
        return true;
    }
  };

  const completedSteps = [
    0 < step && isStepValid(0),
    1 < step && isStepValid(1),
    2 < step && isStepValid(2),
    3 < step && isStepValid(3),
    4 < step && isStepValid(4),
    5 < step && isStepValid(5),
    6 < step && isStepValid(6),
    7 < step && isStepValid(7),
  ];

  const accessibleSteps = [
    true,
    isStepValid(0),
    isStepValid(0) && isStepValid(1),
    isStepValid(0) && isStepValid(1) && isStepValid(2),
    isStepValid(0) && isStepValid(1) && isStepValid(2) && isStepValid(3),
    isStepValid(0) && isStepValid(1) && isStepValid(2) && isStepValid(3) && isStepValid(4),
    isStepValid(0) && isStepValid(1) && isStepValid(2) && isStepValid(3) && isStepValid(4) && isStepValid(5),
    pipelineStatus === "completed" || viewMode === "results",
  ];

  /* -------------------- HANDLERS -------------------- */
  const handleUploaded = (file: File, resp: DatasetUploadResponse) => {
    setUploadedFile(file);
    setDataset(resp);
    setMappingMode("manual");
    setManualMapping(mappingFromDetected(resp));

    // clear run state
    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
    setAutoDownloadedRunId(null);
    setRunLogs([]);
  };

  const handleDatasetUpdate = (resp: DatasetUploadResponse) => {
    setDataset(resp);
    setManualMapping((prev) => {
      const cols = resp.columns ?? [];
      const detected = mappingFromDetected(resp);
      const next = { ...prev };

      if (next.case_id && !cols.includes(next.case_id)) next.case_id = "";
      if (next.activity && !cols.includes(next.activity)) next.activity = "";
      if (next.timestamp && !cols.includes(next.timestamp)) next.timestamp = "";
      if (next.resource && !cols.includes(next.resource)) next.resource = null;

      if (!next.case_id && detected.case_id) next.case_id = detected.case_id;
      if (!next.activity && detected.activity) next.activity = detected.activity;
      if (!next.timestamp && detected.timestamp) next.timestamp = detected.timestamp;
      if (!next.resource && detected.resource) next.resource = detected.resource;

      return next;
    });
    if (customTargetColumn && !resp.columns.includes(customTargetColumn)) {
      setCustomTargetColumn(null);
      if (predictionTask === "custom_activity") {
        setPredictionTask(null);
      }
    }
  };

  const handleSampleDataLoaded = (_file: File, resp: DatasetUploadResponse) => {
    setManualMapping(mappingFromDetected(resp));
  };

  const clearUpload = () => {
    setUploadedFile(null);
    setDataset(null);
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
    setAutoDownloadedRunId(null);
    setRunLogs([]);
  };

  const prepareRerun = () => {
    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
    setAutoDownloadedRunId(null);
    setRunLogs([]);
    setViewMode("wizard");
    setStep(6);
  };

  // Clear explainability immediately when user changes model type (no effects)
  const handleSelectModelType = (v: string) => {
    const nextModel = normalizeModelType(v);
    setModelType(v);

    // Reset configuration when model changes
    setConfigMode(null);
    setTransformerConfig(defaultTransformerConfig);
    setGnnConfig(defaultGnnConfig);
    setExplainabilityConfig(defaultExplainabilityConfig);

    if (!isExplainAllowed(explainMethod, nextModel)) {
      setExplainMethod(null);
    }
  };

  const resetAll = () => {
    setStep(0);

    setUploadedFile(null);
    setDataset(null);
    setDatasetMode(null);
    setSplitConfig({ test_size: 0.1, val_split: 0.11 });
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setModelType(null);
    setPredictionTask(null);
    setPredictionCategory(null);
    setCustomTargetColumn(null);
    setExplainMethod(null);
    setConfigMode(null);
    setTransformerConfig(defaultTransformerConfig);
    setGnnConfig(defaultGnnConfig);
    setExplainabilityConfig(defaultExplainabilityConfig);

    setPipelineStatus("idle");
    setProgress(0);

    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);

    setViewMode("wizard");
  };

  /* -------------------- PIPELINE: create run -------------------- */
  const startPipeline = async () => {
    setRunError(null);
    setArtifacts([]);
    setRunStatus(null);
    setRunId(null);

    if (!dataset) {
      setRunError("No dataset available. Please upload a dataset first.");
      return;
    }
    if (!mappingMode) {
      setRunError("Please configure column mapping first.");
      return;
    }

    const mt = modelTypeNormalized;
    const task = taskNormalized;

    if (!mt) {
      setRunError("Invalid model type. Please re-select Step 2.");
      return;
    }
    if (!task) {
      setRunError("Invalid prediction task. Please re-select Step 4.");
      return;
    }

    const explainToSend = isExplainAllowed(explainMethod, mt) ? explainMethod : null;
    const configToSend =
      mt === "transformer"
        ? configMode === "custom"
          ? transformerConfig
          : defaultTransformerConfig
        : configMode === "custom"
        ? gnnConfig
        : defaultGnnConfig;
    const explainabilityConfigToSend =
      explainToSend && explainToSend !== "none"
        ? explainabilityConfig
        : defaultExplainabilityConfig;

    setPipelineStatus("running");
    setProgress(5);

    try {
      const res = await createRun({
        dataset_id: dataset.dataset_id,
        model_type: mt,
        task,
        config: configToSend,
        explainability_config: explainabilityConfigToSend,
        split: splitConfig,
        explainability: explainToSend,
        target_column: task === "custom_activity" ? customTargetColumn : null,
        mapping_mode: "manual",
        column_mapping: manualMapping,
      });

      setRunId(res.run_id);

      const st = await getRun(res.run_id);
      setRunStatus(st);
      setProgress(st.status === "queued" ? 15 : 40);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRunError(msg);
      setPipelineStatus("idle");
      setProgress(0);
    }
  };

  /* -------------------- PIPELINE: poll status -------------------- */
  useEffect(() => {
    if (pipelineStatus !== "running") return;
    if (!runId) return;

    let cancelled = false;

    const tick = async () => {
      try {
        const st = await getRun(runId);
        if (cancelled) return;

        setRunStatus(st);
        let cleanedLogs: string[] = [];
        try {
          const logs = await getRunLogs(runId, 400);
          const rawLines = logs.lines ?? [];
          cleanedLogs = rawLines
            .map(cleanLogLine)
            .filter((line) => line.length > 0)
            .filter((line) => !isNoiseLine(line));
          if (!cancelled && cleanedLogs.length > 0) setRunLogs(cleanedLogs);
        } catch {
          // Keep last known logs if polling fails.
        }

        if (st.status === "succeeded" || st.status === "failed") {
          setProgress(100);
        } else {
          const nextProgress = estimateProgressFromLogs(cleanedLogs, st);
          setProgress((prev) => Math.max(prev, nextProgress));
        }

        if (st.status === "succeeded") {
          const arts = await listArtifacts(runId);
          if (cancelled) return;
          setArtifacts(arts.artifacts);
          setPipelineStatus("completed");
          setStep(7);
          setViewMode("results");
          if (autoDownloadedRunId !== runId) {
            const link = document.createElement("a");
            link.href = artifactsZipUrl(runId);
            link.download = `run_${runId}_artifacts.zip`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            setAutoDownloadedRunId(runId);
          }
        }

        if (st.status === "failed") {
          setRunError(st.error ?? "Run failed. Check backend logs.txt for details.");
          setPipelineStatus("idle");
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setRunError(msg);
        setPipelineStatus("idle");
      }
    };

    const interval = window.setInterval(tick, 1000);
    tick();

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [autoDownloadedRunId, pipelineStatus, runId]);

  /* -------------------- RENDER -------------------- */
  const showResults = viewMode === "results";

  return (
    <div className="flex h-screen flex-col">
      <div className="shrink-0 border-b border-brand-100 bg-white shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <img
            src={tumLogo}
            alt="TUM"
            className="h-8 w-auto object-contain"
          />
          <img
            src={ppmxLogo}
            alt="PPMX"
            className="h-12 w-auto object-contain"
          />
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <Sidebar currentStep={step} completedSteps={completedSteps} accessibleSteps={accessibleSteps} />

        {showResults ? (
          <ResultsView
            runId={runId}
            onBackToPipeline={() => {
              setViewMode("wizard");
              setStep(6);
            }}
            onRerun={prepareRerun}
            onFinish={resetAll}
          />
        ) : (
          <div className="flex-1 flex flex-col min-w-0 bg-brand-50">
            <div className="px-8 pt-8 shrink-0">
              <StepProgressHeader step={step} totalSteps={TOTAL_STEPS} />
            </div>

            <div className="flex-1 overflow-auto min-w-0">
              <div className="w-full px-8 py-6">
                {step === 0 && (
                  <Step1Upload
                    uploadedFile={uploadedFile}
                    dataset={dataset}
                    onUploaded={handleUploaded}
                    onDatasetUpdate={handleDatasetUpdate}
                    onClear={clearUpload}
                    mode={datasetMode}
                    onModeChange={setDatasetMode}
                    splitConfig={splitConfig}
                    onSplitConfigChange={setSplitConfig}
                    onSampleDataLoaded={handleSampleDataLoaded}
                    manualMapping={manualMapping}
                    onManualMappingChange={(patch) =>
                      setManualMapping((prev) => ({ ...prev, ...patch }))
                    }
                  />
                )}

                {step === 1 && (
                  <Step2Model modelType={modelType} onSelect={handleSelectModelType} />
                )}

                {step === 2 && (
                  <Step5Config
                    modelType={modelTypeNormalized}
                    mode={configMode}
                    onSelect={setConfigMode}
                    transformerConfig={transformerConfig}
                    onTransformerChange={setTransformerConfig}
                    defaultTransformerConfig={defaultTransformerConfig}
                    gnnConfig={gnnConfig}
                    onGnnChange={setGnnConfig}
                    defaultGnnConfig={defaultGnnConfig}
                  />
                )}

                {step === 3 && (
                  <Step3Prediction
                    task={predictionTask}
                    category={predictionCategory}
                    targetColumn={customTargetColumn}
                    dataset={dataset}
                    onSelectTask={(nextTask) => {
                      setPredictionTask(nextTask);
                      if (nextTask === "event_time" || nextTask === "remaining_time") {
                        setPredictionCategory("regression");
                      } else {
                        setPredictionCategory("classification");
                      }
                      if (nextTask !== "custom_activity") {
                        setCustomTargetColumn(null);
                      }
                    }}
                    onSelectCategory={(nextCategory) => {
                      setPredictionCategory(nextCategory);
                      setPredictionTask(null);
                      setCustomTargetColumn(null);
                    }}
                    onTargetColumnChange={setCustomTargetColumn}
                  />
                )}

                {step === 4 && (
                  <Step4Explainability
                    modelType={modelTypeNormalized}
                    method={explainMethod}
                    onSelect={setExplainMethod}
                  />
                )}

                {step === 5 && (
                  <Step5ExplainabilityConfig
                    modelType={modelTypeNormalized}
                    method={explainMethod}
                    config={explainabilityConfig}
                    defaultConfig={defaultExplainabilityConfig}
                    onChange={setExplainabilityConfig}
                  />
                )}

                {step === 6 && (
                  <Step6Review
                    uploadedFile={uploadedFile}
                    dataset={dataset}
                    modelType={modelType}
                    predictionTask={predictionTask}
                    explainMethod={explainMethod} // OK: ExplainValue is a string union
                    mappingMode={mappingMode}
                    manualMapping={manualMapping}
                    configMode={configMode}
                    explainabilityConfig={explainabilityConfig}
                    pipelineStatus={pipelineStatus}
                    progress={progress}
                    runId={runId}
                    runStatus={runStatus}
                    artifacts={artifacts}
                    logs={runLogs}
                    error={runError}
                    onStartPipeline={startPipeline}
                    onRerun={prepareRerun}
                    onFinish={resetAll}
                    onViewResults={() => {
                      setStep(7);
                      setViewMode("results");
                    }}
                  />
                )}
              </div>
            </div>

            <div className="shrink-0 px-8 pb-6 border-t border-brand-100 bg-white">
              <WizardFooter
                step={step}
                canContinue={pipelineStatus !== "running" && isStepValid()}
                onCancel={resetAll}
                onPrevious={() => {
                  if (pipelineStatus === "running") return;
                  prevStep();
                }}
                onContinue={() => {
                  if (pipelineStatus === "running") return;
                  nextStep();
                }}
              />
              <div className="pt-4 text-center text-sm text-gray-500">
                Built with <span className="text-pink-500">❤</span> at TUM
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
