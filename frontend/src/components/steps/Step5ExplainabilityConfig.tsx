import React from "react";
import { SlidersHorizontal } from "lucide-react";

import type { ExplainValue } from "./Step4Explainability";

export type ExplainabilityConfig = {
  local_explanation_samples: number;
  global_explanation_sample_percent: number;
  evaluation_samples: number;
  min_prefix_length: number;
  max_prefix_length: number | null;
  transformer_explanation_samples: number;
  evaluation_sampling_strategy: "evenly_spaced" | "random" | "manual" | "diverse";
  evaluation_random_seed: number;
  evaluation_sample_indices: string;
  evaluation_protocol_name: string;
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
  const disabled = !method || method === "none" || !modelType;
  const cfg = disabled ? defaultConfig : config;
  const isGnn = modelType === "gnn";
  const isTransformer = modelType === "transformer";

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
            <h3 className="text-lg font-semibold">
              {isTransformer ? "Transformer Explainability Config" : "GNN Explainability Config"}
            </h3>
            <p className="text-sm text-gray-600">
              {isTransformer
                ? "Configure deterministic sampling for transformer explanation evaluation."
                : "Configure deterministic sampling and prefix limits for GNN explanation evaluation."}
            </p>

            {disabled ? (
              <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
                {method === "none"
                  ? "Explainability is disabled, so these options will not be used."
                  : "Select a model and explainability method first."}
              </div>
            ) : isGnn ? (
              <div className="mt-4 grid grid-cols-2 gap-4">
                <ParameterField
                  label="Local explanation samples"
                  value={cfg.local_explanation_samples}
                  placeholder="10"
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
                  helpText="Uses this percentage of filtered test graphs for global explainability and evaluation aggregation."
                />
                <ParameterField
                  label="Evaluation samples"
                  value={cfg.evaluation_samples}
                  placeholder="10"
                  min="1"
                  onChange={(e) => update("evaluation_samples", n(e.target.value))}
                  helpText="Number of filtered test graphs used for evaluation metrics. Values above the filtered graph count are clamped automatically."
                />
                <ParameterField
                  label="Minimum prefix length"
                  value={cfg.min_prefix_length}
                  placeholder="5"
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
                <SelectField
                  label="Sampling strategy"
                  value={cfg.evaluation_sampling_strategy}
                  options={[
                    { value: "evenly_spaced", label: "Evenly spaced" },
                    { value: "random", label: "Random seed" },
                    { value: "manual", label: "Manual indices" },
                    { value: "diverse", label: "Diverse coverage" },
                  ]}
                  onChange={(value) => update("evaluation_sampling_strategy", value)}
                  helpText="Controls how filtered test graphs are selected for local, global, and evaluation outputs."
                />
                <ParameterField
                  label="Random seed"
                  value={cfg.evaluation_random_seed}
                  placeholder="42"
                  onChange={(e) => update("evaluation_random_seed", n(e.target.value))}
                  helpText="Used when sampling is random and recorded in the config JSON."
                />
                <TextField
                  label="Selected graph indices"
                  value={cfg.evaluation_sample_indices}
                  placeholder="0, 5, 12"
                  onChange={(value) => update("evaluation_sample_indices", value)}
                  helpText="Optional comma-separated original test graph indices. Used only when sampling strategy is Manual indices."
                />
                <TextField
                  label="Config name"
                  value={cfg.evaluation_protocol_name}
                  placeholder="Perturbation-Based Explainability Evaluation"
                  onChange={(value) => update("evaluation_protocol_name", value)}
                />
              </div>
            ) : isTransformer ? (
              <div className="mt-4 grid grid-cols-2 gap-4">
                <ParameterField
                  label="Local explanation samples"
                  value={cfg.local_explanation_samples}
                  placeholder="10"
                  min="0"
                  onChange={(e) => update("local_explanation_samples", n(e.target.value))}
                  helpText="Number of test sequences for local LIME plots. 0 disables local plots. Values above the filtered sequence count are clamped automatically."
                />
                <ParameterField
                  label="Global explanation sample %"
                  value={cfg.global_explanation_sample_percent}
                  placeholder="100"
                  min="1"
                  max="100"
                  onChange={(e) => update("global_explanation_sample_percent", n(e.target.value))}
                  helpText="Uses this percentage of filtered test sequences for global SHAP explainability."
                />
                <ParameterField
                  label="Evaluation samples"
                  value={cfg.evaluation_samples}
                  placeholder="10"
                  min="1"
                  onChange={(e) => update("evaluation_samples", n(e.target.value))}
                  helpText="Number of filtered test sequences used for evaluation metrics. Values above the filtered sequence count are clamped automatically."
                />
                <ParameterField
                  label="Minimum prefix length"
                  value={cfg.min_prefix_length}
                  placeholder="5"
                  min="1"
                  onChange={(e) => update("min_prefix_length", n(e.target.value))}
                  helpText="Only sequences with at least this many non-padding tokens are considered."
                />
                <ParameterField
                  label="Maximum prefix length"
                  value={cfg.max_prefix_length}
                  placeholder="No maximum"
                  min="1"
                  onChange={(e) =>
                    update("max_prefix_length", e.target.value === "" ? null : n(e.target.value))
                  }
                  helpText="Leave empty to include all longer sequences."
                />
                <ParameterField
                  label="Explanation / evaluation samples (legacy)"
                  value={cfg.transformer_explanation_samples}
                  placeholder="50"
                  min="1"
                  onChange={(e) => update("transformer_explanation_samples", n(e.target.value))}
                  helpText="Fallback sample count when local/global values are not set."
                />
                <SelectField
                  label="Sampling strategy"
                  value={cfg.evaluation_sampling_strategy}
                  options={[
                    { value: "evenly_spaced", label: "Evenly spaced" },
                    { value: "random", label: "Random seed" },
                    { value: "manual", label: "Manual indices" },
                    { value: "diverse", label: "Diverse coverage" },
                  ]}
                  onChange={(value) => update("evaluation_sampling_strategy", value)}
                  helpText="Controls how test sample indices are selected for the evaluation config."
                />
                <ParameterField
                  label="Random seed"
                  value={cfg.evaluation_random_seed}
                  placeholder="42"
                  onChange={(e) => update("evaluation_random_seed", n(e.target.value))}
                  helpText="Used when sampling is random and recorded in the config JSON."
                />
                <TextField
                  label="Selected sample indices"
                  value={cfg.evaluation_sample_indices}
                  placeholder="0, 5, 12"
                  onChange={(value) => update("evaluation_sample_indices", value)}
                  helpText="Optional comma-separated test indices. Used only when sampling strategy is Manual indices."
                />
                <TextField
                  label="Config name"
                  value={cfg.evaluation_protocol_name}
                  placeholder="Perturbation-Based Explainability Evaluation"
                  onChange={(value) => update("evaluation_protocol_name", value)}
                />
              </div>
            ) : null}
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

function SelectField({
  label,
  value,
  options,
  helpText,
  onChange,
}: {
  label: string;
  value: ExplainabilityConfig["evaluation_sampling_strategy"];
  options: Array<{ value: ExplainabilityConfig["evaluation_sampling_strategy"]; label: string }>;
  helpText?: string;
  onChange: (value: ExplainabilityConfig["evaluation_sampling_strategy"]) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ExplainabilityConfig["evaluation_sampling_strategy"])}
        className="w-full bg-white text-black border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {helpText ? <div className="mt-2 text-xs text-gray-500">{helpText}</div> : null}
    </div>
  );
}

function TextField({
  label,
  value,
  placeholder,
  helpText,
  onChange,
}: {
  label: string;
  value: string;
  placeholder: string;
  helpText?: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-white text-black border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500"
      />
      {helpText ? <div className="mt-2 text-xs text-gray-500">{helpText}</div> : null}
    </div>
  );
}
