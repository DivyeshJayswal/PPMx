import React from "react";
import { SlidersHorizontal } from "lucide-react";

import type { ExplainValue } from "./Step4Explainability";

export type ExplainabilityConfig = {
  local_explanation_samples: number;
  global_explanation_sample_percent: number;
  min_prefix_length: number;
  max_prefix_length: number | null;
};

type Step5ExplainabilityConfigProps = {
  modelType: "gnn" | "transformer" | null;
  method: ExplainValue | null;
  config: ExplainabilityConfig;
  defaultConfig: ExplainabilityConfig;
  onChange: (cfg: ExplainabilityConfig) => void;
};

export default function Step5ExplainabilityConfig({
  modelType,
  method,
  config,
  defaultConfig,
  onChange,
}: Step5ExplainabilityConfigProps) {
  const disabled = !method || method === "none" || modelType !== "gnn";
  const cfg = disabled ? defaultConfig : config;

  const update = <K extends keyof ExplainabilityConfig>(
    key: K,
    value: ExplainabilityConfig[K]
  ) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-semibold">Explainability Configuration</h2>
        <p className="text-sm text-brand-600">
          Configure which test-prefix samples are used when explainability is enabled.
        </p>
      </div>

      <div className="border rounded-xl p-6 bg-white">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-brand-500">
            <SlidersHorizontal className="w-5 h-5 text-white" />
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold">GNN Explainability Samples</h3>
            <p className="text-sm text-gray-600">
              Prefix limits filter the test graphs before local and global explainability samples are selected.
            </p>

            {disabled ? (
              <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
                {modelType === "transformer"
                  ? "Transformer explainability uses the method defaults; no GNN prefix sampling configuration is needed."
                  : method === "none"
                  ? "Explainability is disabled, so these options will not be used."
                  : "Select a GNN explainability method first."}
              </div>
            ) : (
              <div className="mt-4 grid grid-cols-2 gap-4">
                <ParameterField
                  label="Local explanation samples"
                  value={cfg.local_explanation_samples}
                  placeholder="5"
                  min="0"
                  onChange={(e) => update("local_explanation_samples", n(e.target.value))}
                  helpText="0 disables local plots. Values above the filtered graph count are clamped automatically."
                />
                <ParameterField
                  label="Global explanation sample %"
                  value={cfg.global_explanation_sample_percent}
                  placeholder="1"
                  min="1"
                  max="100"
                  onChange={(e) => update("global_explanation_sample_percent", n(e.target.value))}
                  helpText="Uses this percentage of filtered test graphs for global explainability and benchmark aggregation."
                />
                <ParameterField
                  label="Minimum prefix length"
                  value={cfg.min_prefix_length}
                  placeholder="1"
                  min="1"
                  onChange={(e) => update("min_prefix_length", n(e.target.value))}
                  helpText="Only prefixes with length at least this value are considered."
                />
                <ParameterField
                  label="Maximum prefix length"
                  value={cfg.max_prefix_length}
                  placeholder="No maximum"
                  min="1"
                  onChange={(e) =>
                    update("max_prefix_length", e.target.value === "" ? null : n(e.target.value))
                  }
                  helpText="Leave empty to include all longer prefixes."
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function n(v: string): number {
  return v === "" ? NaN : Number(v);
}

function ParameterField({
  label,
  value,
  placeholder,
  min,
  max,
  helpText,
  onChange,
}: {
  label: string;
  value: number | null;
  placeholder: string;
  min?: string;
  max?: string;
  helpText?: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <input
        type="number"
        value={typeof value === "number" && Number.isFinite(value) ? value : ""}
        placeholder={placeholder}
        min={min}
        max={max}
        onChange={onChange}
        className="w-full bg-white text-black border rounded px-3 py-2 appearance-none focus:outline-none focus:ring-2 focus:ring-brand-500"
      />
      {helpText ? <div className="mt-2 text-xs text-gray-500">{helpText}</div> : null}
    </div>
  );
}
