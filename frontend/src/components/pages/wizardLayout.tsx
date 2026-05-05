// frontend/src/components/pages/wizardLayout.tsx
import { useEffect, useMemo, useState } from "react";

import Sidebar from "../layout/Sidebar";

import Step1Upload from "../steps/Step1Upload";
import Step2Mapping, { type ManualMapping, type MappingMode } from "../steps/Step2Mapping";
import Step2Model from "../steps/Step2Model";
import Step3Prediction from "../steps/Step3Prediction";
import Step4Explainability, { type ExplainValue } from "../steps/Step4Explainability";
import Step5Config, {
  type ConfigMode,
  type GnnConfig,
  type TransformerConfig,
} from "../steps/Step5Config";
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

const ANSI_RE = /\x1b\[[0-9;]*m/g;

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

function estimateProgressFromLogs(lines: string[], status: RunStatus | null): number {
  if (!status) return 0;
  if (status.status === "queued") return 10;
  if (status.status === "failed") return 100;
  if (status.status === "succeeded") return 100;

  const epoch = extractEpochProgress(lines);
  let progress = 25;

  if (epoch) {
    const frac = Math.min(1, epoch.current / epoch.total);
    progress = 30 + frac * 50; // 30-80
  }

  const joined = lines.join("\n").toLowerCase();
  if (joined.includes("evaluating on test") || joined.includes("evaluating")) {
    progress = Math.max(progress, 85);
  }
  if (joined.includes("saving results") || joined.includes("results saved")) {
    progress = Math.max(progress, 92);
  }
  if (joined.includes("explainability")) {
    progress = Math.max(progress, 95);
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
  const positiveInts = [cfg.hidden, cfg.epochs, cfg.batch_size, cfg.patience].every(
    (v) => Number.isInteger(v) && v > 0
  );

  const dropoutOk =
    typeof cfg.dropout_rate === "number" && cfg.dropout_rate > 0 && cfg.dropout_rate < 1;

  const lrOk = typeof cfg.lr === "number" && cfg.lr > 0;

  return positiveInts && dropoutOk && lrOk;
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
      dropout_rate: 0.1,
      lr: 4e-4,
      epochs: 5,
      batch_size: 64,
      patience: 10,
    }),
    []
  );

  const [transformerConfig, setTransformerConfig] =
    useState<TransformerConfig>(defaultTransformerConfig);
  const [gnnConfig, setGnnConfig] = useState<GnnConfig>(defaultGnnConfig);

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
  const isStepValid = () => {
    switch (step) {
      case 0:
        return dataset !== null && !!dataset.split_paths;
      case 1:
        if (!dataset) return false;
        if (mappingMode === null) return false;
        return validateManualMapping(manualMapping);
      case 2:
        return modelTypeNormalized !== null;
      case 3: {
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
      case 4:
        if (!taskNormalized) return false;
        if (taskNormalized === "custom_activity") {
          if (!customTargetColumn) return false;
          if (!dataset?.column_types) return false;
          if (dataset.column_types[customTargetColumn] !== "categorical") return false;
        }
        return true;
      case 5:
        return explainMethod !== null;
      case 7:
        return true;
      default:
        return true;
    }
  };

  const completedSteps = [
    dataset !== null && !!dataset.split_paths,
    !!dataset && validateManualMapping(manualMapping),
    modelTypeNormalized !== null,
    configMode !== null,
    taskNormalized !== null,
    explainMethod !== null,
    pipelineStatus === "completed",
    viewMode === "results",
  ];

  /* -------------------- HANDLERS -------------------- */
  const handleUploaded = (file: File, resp: DatasetUploadResponse) => {
    setUploadedFile(file);
    setDataset(resp);
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

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
      const next = { ...prev };

      if (next.case_id && !cols.includes(next.case_id)) next.case_id = "";
      if (next.activity && !cols.includes(next.activity)) next.activity = "";
      if (next.timestamp && !cols.includes(next.timestamp)) next.timestamp = "";
      if (next.resource && !cols.includes(next.resource)) next.resource = null;

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
    const dm = resp.detected_mapping ?? {};
    setManualMapping({
      case_id: dm.case_id ?? "",
      activity: dm.activity ?? "",
      timestamp: dm.timestamp ?? "",
      resource: dm.resource ?? null,
    });
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

  // Clear explainability immediately when user changes model type (no effects)
  const handleSelectModelType = (v: string) => {
    const nextModel = normalizeModelType(v);
    setModelType(v);

    // Reset Step 5 config when model changes
    setConfigMode(null);
    setTransformerConfig(defaultTransformerConfig);
    setGnnConfig(defaultGnnConfig);

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
      setRunError("Invalid model type. Please re-select Step 3.");
      return;
    }
    if (!task) {
      setRunError("Invalid prediction task. Please re-select Step 5.");
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

    setPipelineStatus("running");
    setProgress(5);

    try {
      const res = await createRun({
        dataset_id: dataset.dataset_id,
        model_type: mt,
        task,
        config: configToSend,
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
          const logs = await getRunLogs(runId, 120);
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
  }, [pipelineStatus, runId]);

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
        <Sidebar currentStep={step} completedSteps={completedSteps} />

        {showResults ? (
          <ResultsView
            runId={runId}
            onBackToPipeline={() => {
              setViewMode("wizard");
              setStep(6);
            }}
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
                  />
                )}

                {step === 1 && (
                  <Step2Mapping
                    dataset={dataset}
                    manualMapping={manualMapping}
                    onManualMappingChange={(patch) =>
                      setManualMapping((prev) => ({ ...prev, ...patch }))
                    }
                  />
                )}

                {step === 2 && (
                  <Step2Model modelType={modelType} onSelect={handleSelectModelType} />
                )}

                {step === 3 && (
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

                {step === 4 && (
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

                {step === 5 && (
                  <Step4Explainability
                    modelType={modelTypeNormalized}
                    method={explainMethod}
                    onSelect={setExplainMethod}
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
                    pipelineStatus={pipelineStatus}
                    progress={progress}
                    runId={runId}
                    runStatus={runStatus}
                    artifacts={artifacts}
                    logs={runLogs}
                    error={runError}
                    onStartPipeline={startPipeline}
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
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
