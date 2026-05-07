import { useEffect, useRef, useState, type CSSProperties } from "react";
import Card from "../ui/card";
import UploadDropzone from "../ui/UploadDropZone";
import Step2Mapping, { type ManualMapping } from "./Step2Mapping";
import type { DatasetUploadResponse, SampleDatasetInfo, SplitConfig } from "../../lib/api";
import {
  generateSplits,
  listSampleDatasets,
  preprocessedDatasetUrl,
  preprocessDataset,
  sampleDatasetUrl,
  splitDownloadUrl,
  uploadDataset,
  uploadSplitsNewDataset,
  type PreprocessOptions,
} from "../../lib/api";

type Step1UploadProps = {
  uploadedFile: File | null;
  dataset: DatasetUploadResponse | null;
  mode: "raw" | "preprocessed" | "skip" | null;
  splitConfig: SplitConfig;
  onModeChange: (mode: "raw" | "preprocessed" | "skip") => void;
  onSplitConfigChange: (cfg: SplitConfig) => void;
  onUploaded: (file: File, dataset: DatasetUploadResponse) => void;
  onDatasetUpdate?: (dataset: DatasetUploadResponse) => void;
  onClear?: () => void;
  onSampleDataLoaded?: (file: File, dataset: DatasetUploadResponse) => void;
  manualMapping: ManualMapping;
  onManualMappingChange: (patch: Partial<ManualMapping>) => void;
};

const MAX_UPLOAD_MB = 400;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const MIN_SPLIT_PCT = 1;
const TRAIN_COLOR = "#1E3F7C";
const VALID_COLOR = "#3F6DBE";
const TEST_COLOR = "#F47F11";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function getDatasetFormat(name: string): "csv" | "xes" | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".csv")) return "csv";
  if (lower.endsWith(".xes") || lower.endsWith(".xes.gz")) return "xes";
  return null;
}

function getDisplayExt(name: string): string {
  const lower = name.toLowerCase();
  if (lower.endsWith(".xes.gz")) return "xes.gz";
  const i = lower.lastIndexOf(".");
  if (i === -1) return "";
  return lower.slice(i + 1);
}

function validateManualMapping(mapping: ManualMapping): boolean {
  const required = [mapping.case_id, mapping.activity, mapping.timestamp];
  if (required.some((value) => value.trim().length === 0)) return false;
  const selected = [mapping.case_id, mapping.activity, mapping.timestamp, mapping.resource].filter(
    (value): value is string => !!value && value.trim().length > 0
  );
  return new Set(selected).size === selected.length;
}

export default function Step1Upload({
  uploadedFile,
  dataset,
  mode,
  splitConfig,
  onModeChange,
  onSplitConfigChange,
  onUploaded,
  onDatasetUpdate,
  onClear,
  onSampleDataLoaded,
  manualMapping,
  onManualMappingChange,
}: Step1UploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [isGeneratingSplits, setIsGeneratingSplits] = useState(false);
  const [isUploadingSplits, setIsUploadingSplits] = useState(false);
  const [isSampleLoading, setIsSampleLoading] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [sampleDatasets, setSampleDatasets] = useState<SampleDatasetInfo[]>([]);
  const [selectedSample, setSelectedSample] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [preprocessError, setPreprocessError] = useState<string | null>(null);
  const [splitError, setSplitError] = useState<string | null>(null);
  const [splitUploadError, setSplitUploadError] = useState<string | null>(null);

  const [format, setFormat] = useState<"csv" | "xes" | null>(null);
  const [options, setOptions] = useState<PreprocessOptions>({
    sort_and_normalize_timestamps: true,
    check_millisecond_order: true,
    impute_categorical: true,
    impute_numeric_neighbors: true,
    drop_missing_timestamps: true,
    fill_remaining_missing: true,
    remove_duplicates: true,
  });

  const [trainSplit, setTrainSplit] = useState<File | null>(null);
  const [valSplit, setValSplit] = useState<File | null>(null);
  const [testSplit, setTestSplit] = useState<File | null>(null);
  const [dragHandle, setDragHandle] = useState<"train" | "val" | null>(null);
  const sliderRef = useRef<HTMLDivElement | null>(null);

  const calcPercentsFromSplit = (cfg: SplitConfig) => {
    const testPct = Math.round(cfg.test_size * 100);
    const remainPct = 100 - testPct;
    const valPct = Math.round(remainPct * cfg.val_split);
    const trainPct = remainPct - valPct;
    return {
      trainPct,
      valPct,
      testPct,
      trainEnd: trainPct,
      valEnd: trainPct + valPct,
    };
  };

  const initial = calcPercentsFromSplit(splitConfig);
  const [trainEndPct, setTrainEndPct] = useState(initial.trainEnd);
  const [valEndPct, setValEndPct] = useState(initial.valEnd);

  useEffect(() => {
    if (dragHandle) return;
    const next = calcPercentsFromSplit(splitConfig);
    setTrainEndPct(next.trainEnd);
    setValEndPct(next.valEnd);
  }, [splitConfig, dragHandle]);

  const trainPct = trainEndPct;
  const valPct = valEndPct - trainEndPct;
  const testPct = 100 - valEndPct;
  const trainEnd = trainEndPct;
  const valEnd = valEndPct;

  const updateSplitFromPercents = (nextTrainEnd: number, nextValEnd: number) => {
    const clampedTrain = Math.max(MIN_SPLIT_PCT, Math.min(nextTrainEnd, 100 - 2 * MIN_SPLIT_PCT));
    const clampedVal = Math.max(
      clampedTrain + MIN_SPLIT_PCT,
      Math.min(nextValEnd, 100 - MIN_SPLIT_PCT)
    );

    const nextValPct = clampedVal - clampedTrain;
    const nextTestPct = 100 - clampedVal;

    const nextTestSize = nextTestPct / 100;
    const remaining = 1 - nextTestSize;
    const nextValSplit = remaining > 0 ? nextValPct / 100 / remaining : 0.5;

    setTrainEndPct(clampedTrain);
    setValEndPct(clampedVal);
    onSplitConfigChange({
      test_size: Number(nextTestSize.toFixed(4)),
      val_split: Number(nextValSplit.toFixed(4)),
    });
  };

  const percentFromClientX = (clientX: number) => {
    if (!sliderRef.current) return null;
    const rect = sliderRef.current.getBoundingClientRect();
    const raw = ((clientX - rect.left) / rect.width) * 100;
    return Math.max(0, Math.min(100, raw));
  };

  const handlePointerMove = (clientX: number) => {
    const pct = percentFromClientX(clientX);
    if (pct === null || !dragHandle) return;
    if (dragHandle === "train") {
      updateSplitFromPercents(pct, valEndPct);
    } else {
      updateSplitFromPercents(trainEndPct, pct);
    }
  };

  useEffect(() => {
    setPreprocessError(null);
    setSplitError(null);
    setIsPreprocessing(false);
    setIsGeneratingSplits(false);
    setOptions({
      sort_and_normalize_timestamps: true,
      check_millisecond_order: true,
      impute_categorical: true,
      impute_numeric_neighbors: true,
      drop_missing_timestamps: true,
      fill_remaining_missing: true,
      remove_duplicates: true,
    });
  }, [dataset?.dataset_id]);

  useEffect(() => {
    setError(null);
    setPreprocessError(null);
    setSplitError(null);
    setSplitUploadError(null);
    setFormat(null);
    setTrainSplit(null);
    setValSplit(null);
    setTestSplit(null);
  }, [mode]);

  useEffect(() => {
    let cancelled = false;

    const loadSamples = async () => {
      try {
        const items = await listSampleDatasets();
        if (cancelled) return;
        setSampleDatasets(items);
        setSelectedSample((prev) => {
          if (prev && items.some((item) => item.name === prev)) return prev;
          return items[0]?.name ?? "";
        });
      } catch (e) {
        if (cancelled) return;
        setSampleError(e instanceof Error ? e.message : "Failed to load sample dataset list.");
      }
    };

    loadSamples();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleClear = () => {
    setError(null);
    setPreprocessError(null);
    setSplitError(null);
    setSplitUploadError(null);
    setFormat(null);
    setTrainSplit(null);
    setValSplit(null);
    setTestSplit(null);
    onClear?.();
  };

  const handleModeSelect = (nextMode: "raw" | "preprocessed" | "skip") => {
    if (mode !== nextMode) {
      handleClear();
      onModeChange(nextMode);
    }
  };

  const handleUpload = async (file: File) => {
    setError(null);
    setPreprocessError(null);
    setSplitError(null);

    if (!format) {
      setError("Please select a file format first.");
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`File too large. Max allowed size is ${MAX_UPLOAD_MB} MB.`);
      return;
    }

    const detectedFormat = getDatasetFormat(file.name);
    if (!detectedFormat) {
      setError("Unsupported format. Please upload a CSV, XES, or XES.GZ file.");
      return;
    }
    if (detectedFormat !== format) {
      setError(`Selected format is ${format.toUpperCase()}, but you uploaded a .${getDisplayExt(file.name)} file.`);
      return;
    }

    setIsUploading(true);
    try {
      const resp = await uploadDataset(file, { preprocessed: mode === "preprocessed" });
      onUploaded(file, resp);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setError(msg);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSampleData = async () => {
    setSampleError(null);
    if (!selectedSample) {
      setSampleError("Select a sample dataset first.");
      return;
    }
    if (!sampleDatasets.some((item) => item.name === selectedSample && item.configured)) {
      setSampleError("Google Drive link is not configured yet for the selected sample dataset.");
      return;
    }
    setIsSampleLoading(true);
    try {
      const response = await fetch(sampleDatasetUrl(selectedSample));
      if (!response.ok) throw new Error("Could not fetch sample dataset from server.");
      const blob = await response.blob();
      const type = selectedSample.toLowerCase().endsWith(".csv")
        ? "text/csv"
        : "application/octet-stream";
      const file = new File([blob], selectedSample, { type });

      const uploaded = await uploadDataset(file, { preprocessed: false });
      const preprocessed = await preprocessDataset(uploaded.dataset_id, {
        sort_and_normalize_timestamps: true,
        check_millisecond_order: true,
        impute_categorical: true,
        impute_numeric_neighbors: true,
        drop_missing_timestamps: true,
        fill_remaining_missing: true,
        remove_duplicates: true,
      });
      const withSplits = await generateSplits(preprocessed.dataset_id, splitConfig);

      onModeChange("raw");
      onUploaded(file, withSplits);
      onSampleDataLoaded?.(file, withSplits);
    } catch (e) {
      setSampleError(e instanceof Error ? e.message : "Failed to load sample data.");
    } finally {
      setIsSampleLoading(false);
    }
  };

  const handlePreprocess = async () => {
    if (!dataset) return;
    setPreprocessError(null);
    setIsPreprocessing(true);
    try {
      const resp = await preprocessDataset(dataset.dataset_id, options);
      onDatasetUpdate?.(resp);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Preprocessing failed.";
      setPreprocessError(msg);
    } finally {
      setIsPreprocessing(false);
    }
  };

  const handleGenerateSplits = async () => {
    if (!dataset) return;
    if (mode === "raw" && !dataset.is_preprocessed) {
      setSplitError("Please run preprocessing before generating splits.");
      return;
    }
    setSplitError(null);
    setIsGeneratingSplits(true);
    try {
      const resp = await generateSplits(dataset.dataset_id, splitConfig);
      onDatasetUpdate?.(resp);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Split generation failed.";
      setSplitError(msg);
    } finally {
      setIsGeneratingSplits(false);
    }
  };

  const handleUploadSplits = async () => {
    setSplitUploadError(null);
    if (!trainSplit || !valSplit || !testSplit) {
      setSplitUploadError("Please provide train, validation, and test CSV files.");
      return;
    }
    setIsUploadingSplits(true);
    try {
      const resp = await uploadSplitsNewDataset(trainSplit, valSplit, testSplit);
      onDatasetUpdate?.(resp);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Split upload failed.";
      setSplitUploadError(msg);
    } finally {
      setIsUploadingSplits(false);
    }
  };

  const showSplitControls =
    !!dataset &&
    (mode === "preprocessed" || (mode === "raw" && dataset.is_preprocessed));
  const mappingComplete = validateManualMapping(manualMapping);

  return (
    <div className="space-y-8 w-full">
      <div>
        <h2 className="text-2xl font-semibold">Dataset Setup</h2>
        <p className="text-sm text-brand-600">
          Choose how you want to provide data: raw upload, preprocessed upload, or your own splits.
        </p>
      </div>

      <div className="border border-brand-300 bg-brand-50 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-brand-800">Try A Sample Dataset</div>
          <div className="text-xs text-brand-600 mt-0.5">
            Choose any raw dataset from the list.
          </div>
        </div>
        <div className="flex flex-col items-start sm:items-end gap-2 shrink-0 sm:min-w-80">
          <select
            value={selectedSample}
            onChange={(e) => setSelectedSample(e.target.value)}
            disabled={isSampleLoading || !!dataset || sampleDatasets.length === 0}
            className="w-full rounded-md border border-brand-200 bg-white px-3 py-2 text-sm text-gray-800 disabled:bg-gray-100 disabled:text-gray-400"
          >
            {sampleDatasets.length === 0 ? (
              <option value="">No sample datasets available</option>
            ) : (
              sampleDatasets.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                  {item.size_bytes ? ` (${item.format.toUpperCase()}, ${formatBytes(item.size_bytes)})` : ` (${item.format.toUpperCase()})`}
                  {!item.configured ? " [link pending]" : ""}
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            onClick={handleSampleData}
            disabled={
              isSampleLoading ||
              !!dataset ||
              !selectedSample ||
              !sampleDatasets.some((item) => item.name === selectedSample && item.configured)
            }
            className={[
              "px-4 py-2 rounded-md text-sm font-medium border transition w-full sm:w-auto",
              isSampleLoading ||
              !!dataset ||
              !selectedSample ||
              !sampleDatasets.some((item) => item.name === selectedSample && item.configured)
                ? "border-gray-300 bg-gray-100 text-gray-400 cursor-not-allowed"
                : "border-brand-500 bg-brand-600 text-white hover:bg-brand-700",
            ].join(" ")}
          >
            {isSampleLoading ? "Loading sample..." : "Load Sample Dataset"}
          </button>
          {sampleError && (
            <div className="text-xs text-red-600 max-w-xs text-right">{sampleError}</div>
          )}
        </div>
      </div>

      <Card>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            type="button"
            onClick={() => handleModeSelect("raw")}
            className={[
              "border rounded-lg p-4 text-left transition",
              mode === "raw" ? "border-brand-500 ring-1 ring-brand-200" : "hover:border-gray-300",
            ].join(" ")}
          >
            <div className="font-semibold text-gray-900">1) Upload Raw Dataset</div>
            <div className="text-xs text-gray-600 mt-1">CSV or XES + preprocessing options.</div>
          </button>
          <button
            type="button"
            onClick={() => handleModeSelect("preprocessed")}
            className={[
              "border rounded-lg p-4 text-left transition",
              mode === "preprocessed"
                ? "border-brand-500 ring-1 ring-brand-200"
                : "hover:border-gray-300",
            ].join(" ")}
          >
            <div className="font-semibold text-gray-900">2) Upload Preprocessed Dataset</div>
            <div className="text-xs text-gray-600 mt-1">CSV only, skip preprocessing.</div>
          </button>
          <button
            type="button"
            onClick={() => handleModeSelect("skip")}
            className={[
              "border rounded-lg p-4 text-left transition",
              mode === "skip"
                ? "border-brand-500 ring-1 ring-brand-200"
                : "hover:border-gray-300",
            ].join(" ")}
          >
            <div className="font-semibold text-gray-900">3) Skip (Upload Splits)</div>
            <div className="text-xs text-gray-600 mt-1">Upload train/val/test CSVs directly.</div>
          </button>
        </div>
      </Card>

      {mode && (
        <Card>
          <div className="w-full space-y-6">
            {(mode === "raw" || mode === "preprocessed") && !dataset && (
              <div className="space-y-3">
                <div className="border rounded-lg bg-white p-4">
                  <div className="text-sm font-medium text-gray-900">Select file format</div>
                  <div className="mt-3 flex flex-wrap gap-3">
                    <button
                      type="button"
                      className={[
                        "px-4 py-2 rounded-md border text-sm",
                        format === "csv"
                          ? "border-brand-500 bg-brand-50"
                          : "border-gray-200 bg-white hover:bg-gray-50",
                      ].join(" ")}
                      onClick={() => {
                        setError(null);
                        setFormat("csv");
                      }}
                    >
                      CSV
                    </button>
                    {mode === "raw" && (
                      <button
                        type="button"
                        className={[
                          "px-4 py-2 rounded-md border text-sm",
                          format === "xes"
                            ? "border-brand-500 bg-brand-50"
                            : "border-gray-200 bg-white hover:bg-gray-50",
                        ].join(" ")}
                        onClick={() => {
                          setError(null);
                          setFormat("xes");
                        }}
                      >
                        XES
                      </button>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-gray-600">
                    {mode === "preprocessed"
                      ? "Preprocessed uploads are CSV only."
                      : "Choose CSV if you already exported a table, or XES for event logs."}
                  </div>
                </div>

                <div className="border rounded-lg bg-white p-4">
                  <div className="text-sm font-medium text-gray-900">Upload dataset</div>
                  <div className="mt-3">
                    <UploadDropzone
                      onFileSelect={handleUpload}
                      accept={format === "csv" ? ".csv" : format === "xes" ? ".xes,.xes.gz" : ".csv,.xes,.xes.gz"}
                      disabled={!format}
                    />
                  </div>
                </div>

                {isUploading && (
                  <div className="text-sm text-gray-600">Uploading and parsing dataset...</div>
                )}

                {error && (
                  <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                    <div className="font-medium">Upload blocked / failed</div>
                    <div className="mt-1 break-words">{error}</div>
                  </div>
                )}
              </div>
            )}

            {mode === "skip" && !dataset && (
              <div className="space-y-3">
                <div className="border rounded-lg bg-white p-4">
                  <div className="text-sm font-medium text-gray-900">Upload Train CSV</div>
                  <div className="mt-3">
                    <UploadDropzone
                      onFileSelect={(f) => setTrainSplit(f)}
                      accept=".csv"
                      disabled={isUploadingSplits}
                    />
                  </div>
                </div>
                <div className="border rounded-lg bg-white p-4">
                  <div className="text-sm font-medium text-gray-900">Upload Validation CSV</div>
                  <div className="mt-3">
                    <UploadDropzone
                      onFileSelect={(f) => setValSplit(f)}
                      accept=".csv"
                      disabled={isUploadingSplits}
                    />
                  </div>
                </div>
                <div className="border rounded-lg bg-white p-4">
                  <div className="text-sm font-medium text-gray-900">Upload Test CSV</div>
                  <div className="mt-3">
                    <UploadDropzone
                      onFileSelect={(f) => setTestSplit(f)}
                      accept=".csv"
                      disabled={isUploadingSplits}
                    />
                  </div>
                </div>

                <div className="flex flex-wrap gap-3 items-center">
                  <button
                    type="button"
                    onClick={handleUploadSplits}
                    disabled={isUploadingSplits}
                    className={[
                      "px-4 py-2 rounded-md text-sm border",
                      isUploadingSplits
                        ? "border-gray-300 bg-gray-100 text-gray-500"
                        : "border-brand-400 bg-brand-50 text-brand-800 hover:bg-brand-100",
                    ].join(" ")}
                  >
                    {isUploadingSplits ? "Uploading splits..." : "Upload splits"}
                  </button>
                  <div className="text-xs text-gray-600">
                    CSV only. Ensure all three files share the same columns.
                  </div>
                </div>

                {splitUploadError && (
                  <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                    <div className="font-medium">Split upload failed</div>
                    <div className="mt-1 break-words">{splitUploadError}</div>
                  </div>
                )}
              </div>
            )}

            {!!dataset && (
              <div className="w-full border border-green-300 bg-green-50 rounded-lg p-6 space-y-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="font-medium text-green-800">Dataset Ready</div>
                    <div className="text-sm text-green-700 mt-1">
                      {uploadedFile
                        ? `${uploadedFile.name} (${formatBytes(uploadedFile.size)}) — `
                        : ""}
                      {dataset.num_events.toLocaleString()} events,{" "}
                      {dataset.num_cases.toLocaleString()} cases
                    </div>
                    <div className="text-xs text-green-700 mt-2">
                      <span className="font-medium">dataset_id:</span>{" "}
                      <span className="font-mono">{dataset.dataset_id}</span>
                    </div>
                  </div>

                  {onClear && (
                    <button
                      type="button"
                      onClick={handleClear}
                      className="text-sm px-3 py-2 rounded-md border border-green-300 bg-white hover:bg-green-100"
                    >
                      Upload another file
                    </button>
                  )}
                </div>

                <Step2Mapping
                  dataset={dataset}
                  manualMapping={manualMapping}
                  onManualMappingChange={onManualMappingChange}
                  embedded
                />

                {!mappingComplete && (
                  <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
                    Complete the column mapping above before preprocessing or generating dataset splits.
                  </div>
                )}

                {mode === "raw" && mappingComplete && (
                  <div className="bg-white border border-green-200 rounded-md p-4 space-y-3">
                    <div className="font-medium text-gray-800">Preprocessing Options</div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-gray-700">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.sort_and_normalize_timestamps}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              sort_and_normalize_timestamps: e.target.checked,
                            }))
                          }
                        />
                        Sort and normalize timestamps
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.check_millisecond_order}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              check_millisecond_order: e.target.checked,
                            }))
                          }
                        />
                        Check millisecond ordering
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.impute_categorical}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              impute_categorical: e.target.checked,
                            }))
                          }
                        />
                        Impute categorical values
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.impute_numeric_neighbors}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              impute_numeric_neighbors: e.target.checked,
                            }))
                          }
                        />
                        Impute numeric values (neighbors)
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.drop_missing_timestamps}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              drop_missing_timestamps: e.target.checked,
                            }))
                          }
                        />
                        Drop rows missing timestamps
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.fill_remaining_missing}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              fill_remaining_missing: e.target.checked,
                            }))
                          }
                        />
                        Fill remaining missing values
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!options.remove_duplicates}
                          onChange={(e) =>
                            setOptions((prev) => ({
                              ...prev,
                              remove_duplicates: e.target.checked,
                            }))
                          }
                        />
                        Remove duplicate rows
                      </label>
                    </div>

                    <div className="flex flex-wrap gap-3 items-center">
                      <button
                        type="button"
                        onClick={handlePreprocess}
                        disabled={isPreprocessing}
                        className={[
                          "px-4 py-2 rounded-md text-sm border",
                          isPreprocessing
                            ? "border-gray-300 bg-gray-100 text-gray-500"
                            : "border-green-400 bg-green-100 text-green-900 hover:bg-green-200",
                        ].join(" ")}
                      >
                        {isPreprocessing ? "Preprocessing..." : "Run preprocessing"}
                      </button>

                      {dataset.is_preprocessed && (
                        <a
                          href={preprocessedDatasetUrl(dataset.dataset_id)}
                          className="px-4 py-2 rounded-md text-sm border border-brand-300 bg-brand-50 text-brand-700 hover:bg-brand-100"
                          download
                        >
                          Export preprocessed dataset
                        </a>
                      )}
                    </div>

                    {preprocessError && (
                      <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                        <div className="font-medium">Preprocessing failed</div>
                        <div className="mt-1 break-words">{preprocessError}</div>
                      </div>
                    )}
                  </div>
                )}

                {showSplitControls && mappingComplete && (
                  <div className="bg-white border border-green-200 rounded-md p-4 space-y-3">
                    <div className="font-medium text-gray-800">Generate Splits</div>
                    <div className="space-y-4 text-sm text-gray-700">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-brand-700">
                          Train {trainPct.toFixed(1)}%
                        </span>
                        <span className="font-medium text-brand-600">
                          Valid {valPct.toFixed(1)}%
                        </span>
                        <span className="font-medium text-accent-700">
                          Test {testPct.toFixed(1)}%
                        </span>
                      </div>

                      <div
                        ref={sliderRef}
                        className="relative h-10"
                        onPointerMove={(e) => handlePointerMove(e.clientX)}
                        onPointerUp={() => setDragHandle(null)}
                        onPointerLeave={() => setDragHandle(null)}
                      >
                        <div
                          className="absolute top-1/2 -translate-y-1/2 h-2 w-full rounded-full"
                          style={{
                            background: `linear-gradient(to right, ${TRAIN_COLOR} 0% ${trainEnd}%, ${VALID_COLOR} ${trainEnd}% ${valEnd}%, ${TEST_COLOR} ${valEnd}% 100%)`,
                          }}
                        />
                        <div
                          role="slider"
                          aria-valuemin={MIN_SPLIT_PCT}
                          aria-valuemax={100 - 2 * MIN_SPLIT_PCT}
                          aria-valuenow={Math.round(trainEnd)}
                          className="split-handle"
                          style={
                            {
                              left: `${trainEnd}%`,
                              "--handle-color": TRAIN_COLOR,
                              zIndex: 30,
                            } as CSSProperties
                          }
                          onPointerDown={(e) => {
                            e.currentTarget.setPointerCapture(e.pointerId);
                            setDragHandle("train");
                          }}
                        />
                        <div
                          role="slider"
                          aria-valuemin={2 * MIN_SPLIT_PCT}
                          aria-valuemax={100 - MIN_SPLIT_PCT}
                          aria-valuenow={Math.round(valEnd)}
                          className="split-handle"
                          style={
                            {
                              left: `${valEnd}%`,
                              "--handle-color": TEST_COLOR,
                              zIndex: 40,
                            } as CSSProperties
                          }
                          onPointerDown={(e) => {
                            e.currentTarget.setPointerCapture(e.pointerId);
                            setDragHandle("val");
                          }}
                        />
                      </div>
                      <div className="text-xs text-gray-500">
                        Drag the sliders to set Train/Valid/Test splits. Total always equals 100%.
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-3 items-center">
                      <button
                        type="button"
                        onClick={handleGenerateSplits}
                        disabled={isGeneratingSplits}
                        className={[
                          "px-4 py-2 rounded-md text-sm border",
                          isGeneratingSplits
                            ? "border-gray-300 bg-gray-100 text-gray-500"
                            : "border-brand-400 bg-brand-50 text-brand-800 hover:bg-brand-100",
                        ].join(" ")}
                      >
                        {isGeneratingSplits ? "Generating splits..." : "Generate splits"}
                      </button>
                    </div>

                    {splitError && (
                      <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                        <div className="font-medium">Split generation failed</div>
                        <div className="mt-1 break-words">{splitError}</div>
                      </div>
                    )}
                  </div>
                )}

                {dataset.split_paths && (
                  <div className="bg-white border border-green-200 rounded-md p-4 space-y-3">
                    <div className="font-medium text-gray-800">Download Splits</div>
                    <div className="flex flex-wrap gap-3">
                      <a
                        className="px-4 py-2 rounded-md text-sm border border-brand-300 bg-brand-50 text-brand-700 hover:bg-brand-100"
                        href={splitDownloadUrl(dataset.dataset_id, "train")}
                        download
                      >
                        Download train.csv
                      </a>
                      <a
                        className="px-4 py-2 rounded-md text-sm border border-brand-300 bg-brand-50 text-brand-700 hover:bg-brand-100"
                        href={splitDownloadUrl(dataset.dataset_id, "val")}
                        download
                      >
                        Download val.csv
                      </a>
                      <a
                        className="px-4 py-2 rounded-md text-sm border border-brand-300 bg-brand-50 text-brand-700 hover:bg-brand-100"
                        href={splitDownloadUrl(dataset.dataset_id, "test")}
                        download
                      >
                        Download test.csv
                      </a>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>
      )}

      <Card title="Dataset Requirements">
        <div className="bg-brand-50 border border-brand-200 rounded-lg p-4">
          <ul className="text-sm text-gray-700 list-disc ml-5 space-y-1">
            <li>File size: Maximum 400 MB</li>
            <li>Formats: CSV or XES (preprocessed and splits must be CSV)</li>
            <li>Required columns: Case ID, Activity, Timestamp</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}


