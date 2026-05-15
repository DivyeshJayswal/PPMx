import Card from "../ui/card";
import type { ColumnDiagnostic, DatasetUploadResponse } from "../../lib/api";

export type MappingMode = "manual";

export type ManualMapping = {
  case_id: string;
  activity: string;
  timestamp: string;
  resource: string | null;
};

type Step2MappingProps = {
  dataset: DatasetUploadResponse | null;
  manualMapping: ManualMapping;
  onManualMappingChange: (patch: Partial<ManualMapping>) => void;
  embedded?: boolean;
};

function uniqueNonEmpty(values: Array<string | null>): boolean {
  const filtered = values.filter((v): v is string => !!v && v.trim().length > 0);
  return new Set(filtered).size === filtered.length;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function getDiagnostic(
  dataset: DatasetUploadResponse | null,
  column: string
): ColumnDiagnostic | null {
  if (!dataset || !column.trim()) return null;
  return dataset.column_diagnostics?.[column] ?? null;
}

function buildMappingWarnings(
  dataset: DatasetUploadResponse | null,
  manualMapping: ManualMapping
): string[] {
  if (!dataset) return [];

  const warnings: string[] = [];
  const detectedCase = dataset.detected_mapping?.case_id;
  const detectedTimestamp = dataset.detected_mapping?.timestamp;

  const caseDiag = getDiagnostic(dataset, manualMapping.case_id);
  if (manualMapping.case_id && caseDiag) {
    if (caseDiag.looks_event_unique) {
      let message =
        `"${manualMapping.case_id}" looks event-unique: ` +
        `${caseDiag.unique_count.toLocaleString()} distinct values across ` +
        `${dataset.num_events.toLocaleString()} rows. Using it as Case ID may make every event its own case.`;
      if (detectedCase && detectedCase !== manualMapping.case_id) {
        message += ` Suggested case column: "${detectedCase}".`;
      }
      warnings.push(message);
    }
    if (
      caseDiag.timestamp_parse_ratio >= 0.8 &&
      caseDiag.looks_timestamp_like &&
      manualMapping.case_id !== manualMapping.timestamp
    ) {
      warnings.push(
        `"${manualMapping.case_id}" also looks timestamp-like (${formatPercent(
          caseDiag.timestamp_parse_ratio
        )} parseable as dates). It may not be a real case identifier.`
      );
    }
  }

  const timestampDiag = getDiagnostic(dataset, manualMapping.timestamp);
  if (manualMapping.timestamp && timestampDiag) {
    if (timestampDiag.timestamp_parse_ratio < 0.8) {
      let message =
        `"${manualMapping.timestamp}" does not look like a reliable timestamp column ` +
        `(${formatPercent(timestampDiag.timestamp_parse_ratio)} parseable as dates).`;
      if (detectedTimestamp && detectedTimestamp !== manualMapping.timestamp) {
        message += ` Suggested timestamp column: "${detectedTimestamp}".`;
      }
      warnings.push(message);
    }
    if (timestampDiag.max_frequency_share >= 0.95) {
      warnings.push(
        `"${manualMapping.timestamp}" is almost constant across the dataset. That is unusual for an event timestamp column.`
      );
    }
  }

  return warnings;
}

export default function Step2Mapping({
  dataset,
  manualMapping,
  onManualMappingChange,
  embedded = false,
}: Step2MappingProps) {
  const columns = (dataset?.columns ?? []).filter((c) => c !== "__split");
  const canShowManual = !!dataset;

  const manualOk =
    manualMapping.case_id.trim().length > 0 &&
    manualMapping.activity.trim().length > 0 &&
    manualMapping.timestamp.trim().length > 0 &&
    uniqueNonEmpty([
      manualMapping.case_id,
      manualMapping.activity,
      manualMapping.timestamp,
      manualMapping.resource,
    ]);

  const mappingWarnings = buildMappingWarnings(dataset, manualMapping);

  const mappingCard = !dataset ? (
    <Card>
      <div className="p-6 text-sm text-gray-700">Upload a dataset in Step 1 first.</div>
    </Card>
  ) : (
    <div className="space-y-4">
      {canShowManual && (
        <Card title={embedded ? "Column Mapping" : "Manual Mapping"}>
          <div className="space-y-4">
            <MappingSelect
              label="Case ID column"
              value={manualMapping.case_id}
              columns={["", ...columns]}
              onChange={(v) => onManualMappingChange({ case_id: v })}
              placeholder="Select..."
            />
            <MappingSelect
              label="Activity column"
              value={manualMapping.activity}
              columns={["", ...columns]}
              onChange={(v) => onManualMappingChange({ activity: v })}
              placeholder="Select..."
            />
            <MappingSelect
              label="Timestamp column"
              value={manualMapping.timestamp}
              columns={["", ...columns]}
              onChange={(v) => onManualMappingChange({ timestamp: v })}
              placeholder="Select..."
            />
            <MappingSelect
              label="Resource column (optional)"
              value={manualMapping.resource ?? ""}
              columns={["", ...columns]}
              onChange={(v) => onManualMappingChange({ resource: v.trim() ? v : null })}
              placeholder="None"
            />

            {!manualOk && (
              <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                Select Case ID, Activity, and Timestamp columns (must be different). Resource is
                optional.
              </div>
            )}

            {mappingWarnings.length > 0 && (
              <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3 space-y-2">
                <div className="font-medium">Mapping warnings</div>
                <ul className="list-disc ml-5 space-y-1">
                  {mappingWarnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );

  if (embedded) {
    return mappingCard;
  }

  return (
    <div className="space-y-8 w-full">
      <div>
        <h2 className="text-2xl font-semibold">Column Mapping</h2>
        <p className="text-sm text-brand-600">
          Map your dataset columns to the required schema. Resource is optional.
        </p>
      </div>

      {mappingCard}
    </div>
  );
}

function MappingSelect({
  label,
  value,
  columns,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  columns: string[];
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
      <div className="text-sm font-medium text-gray-800">{label}</div>
      <div className="sm:col-span-2">
        <select
          className="w-full border rounded-md px-3 py-2 bg-white text-sm"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {columns.map((c) => (
            <option key={c || "__none"} value={c}>
              {c || placeholder || "Select..."}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

