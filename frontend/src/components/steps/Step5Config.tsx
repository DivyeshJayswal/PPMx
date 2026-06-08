import React from "react";
import { Settings } from "lucide-react";

export type ConfigMode = "default" | "custom";

// Must match backend `default_transformer_config()` keys/types in `ppm_pipeline.py`.
export type TransformerConfig = {
  max_len: number;
  d_model: number;
  num_heads: number;
  num_blocks: number;
  dropout_rate: number;
  epochs: number;
  batch_size: number;
  patience: number;
};

// Must match backend `default_gnn_config()` keys/types in `ppm_pipeline.py`.
export type GnnConfig = {
  hidden: number;
  heads: number;
  num_layers: number;
  dropout_rate: number;
  lr: number;
  epochs: number;
  batch_size: number;
  patience: number;
};

type Step5ConfigProps = {
  modelType: "gnn" | "transformer" | null;
  mode: ConfigMode | null;
  onSelect: (mode: ConfigMode) => void;

  transformerConfig: TransformerConfig;
  onTransformerChange: (cfg: TransformerConfig) => void;
  defaultTransformerConfig: TransformerConfig;

  gnnConfig: GnnConfig;
  onGnnChange: (cfg: GnnConfig) => void;
  defaultGnnConfig: GnnConfig;
};

export default function Step5Config({
  modelType,
  mode,
  onSelect,
  transformerConfig,
  onTransformerChange,
  defaultTransformerConfig,
  gnnConfig,
  onGnnChange,
  defaultGnnConfig,
}: Step5ConfigProps) {
  if (!modelType) {
    return (
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">Select a model type in Step 2 first.</p>
      </div>
    );
  }

  const isTransformer = modelType === "transformer";

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">
          {isTransformer
            ? "These options map 1:1 to the backend Transformer config."
            : "These options map 1:1 to the backend GNN config."}
        </p>
      </div>

      <ConfigCard
        title="Default Configuration"
        description="Use backend defaults."
        selected={mode === "default"}
        onClick={() => onSelect("default")}
      >
        <ParameterGrid
          modelType={modelType}
          editable={false}
          transformerConfig={transformerConfig}
          onTransformerChange={onTransformerChange}
          defaultTransformerConfig={defaultTransformerConfig}
          gnnConfig={gnnConfig}
          onGnnChange={onGnnChange}
          defaultGnnConfig={defaultGnnConfig}
        />
      </ConfigCard>

      <ConfigCard
        title="Custom Configuration"
        description="Override backend defaults."
        selected={mode === "custom"}
        onClick={() => onSelect("custom")}
      >
        <ParameterGrid
          modelType={modelType}
          editable={mode === "custom"}
          transformerConfig={transformerConfig}
          onTransformerChange={onTransformerChange}
          defaultTransformerConfig={defaultTransformerConfig}
          gnnConfig={gnnConfig}
          onGnnChange={onGnnChange}
          defaultGnnConfig={defaultGnnConfig}
        />
      </ConfigCard>
    </div>
  );
}

function ConfigCard({
  title,
  description,
  selected,
  onClick,
  children,
}: {
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      onClick={onClick}
      className={`border rounded-xl p-6 cursor-pointer transition ${
        selected ? "border-brand-500 bg-brand-50" : "border-gray-200 bg-white hover:border-gray-300"
      }`}
    >
      <div className="flex items-start gap-4">
        <Icon selected={selected} />

        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold">{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
          <div className="mt-4">{children}</div>
        </div>

        <Radio selected={selected} />
      </div>
    </div>
  );
}

function ParameterGrid({
  modelType,
  editable,
  transformerConfig,
  onTransformerChange,
  defaultTransformerConfig,
  gnnConfig,
  onGnnChange,
  defaultGnnConfig,
}: {
  modelType: "gnn" | "transformer";
  editable: boolean;
  transformerConfig: TransformerConfig;
  onTransformerChange: (cfg: TransformerConfig) => void;
  defaultTransformerConfig: TransformerConfig;
  gnnConfig: GnnConfig;
  onGnnChange: (cfg: GnnConfig) => void;
  defaultGnnConfig: GnnConfig;
}) {
  if (modelType === "transformer") {
    const cfg = editable ? transformerConfig : defaultTransformerConfig;
    const update = <K extends keyof TransformerConfig>(key: K, value: number) => {
      onTransformerChange({ ...transformerConfig, [key]: value });
    };

    return (
      <div className="grid grid-cols-2 gap-4">
        <ParameterField
          label="Max sequence length"
          value={cfg.max_len}
          placeholder="16"
          editable={editable}
          onChange={(e) => update("max_len", n(e.target.value))}
        />
        <ParameterField
          label="Model dimension"
          value={cfg.d_model}
          placeholder="64"
          editable={editable}
          onChange={(e) => update("d_model", n(e.target.value))}
        />
        <ParameterField
          label="Number of attention heads"
          value={cfg.num_heads}
          placeholder="4"
          editable={editable}
          onChange={(e) => update("num_heads", n(e.target.value))}
        />
        <ParameterField
          label="Number of transformer blocks"
          value={cfg.num_blocks}
          placeholder="2"
          editable={editable}
          onChange={(e) => update("num_blocks", n(e.target.value))}
        />
        <ParameterField
          label="Dropout rate"
          value={cfg.dropout_rate}
          placeholder="0.1"
          editable={editable}
          min="0"
          max="1"
          step="0.05"
          onChange={(e) => update("dropout_rate", n(e.target.value))}
        />
        <ParameterField
          label="Number of epochs"
          value={cfg.epochs}
          placeholder="5"
          editable={editable}
          onChange={(e) => update("epochs", n(e.target.value))}
        />
        <ParameterField
          label="Batch size"
          value={cfg.batch_size}
          placeholder="128"
          editable={editable}
          onChange={(e) => update("batch_size", n(e.target.value))}
        />
        <ParameterField
          label="Early stopping patience"
          value={cfg.patience}
          placeholder="10"
          editable={editable}
          onChange={(e) => update("patience", n(e.target.value))}
        />
      </div>
    );
  }

  const cfg = editable ? gnnConfig : defaultGnnConfig;
  const update = <K extends keyof GnnConfig>(key: K, value: number) => {
    onGnnChange({ ...gnnConfig, [key]: value });
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <ParameterField
        label="Hidden channels"
        value={cfg.hidden}
        placeholder="64"
        editable={editable}
        onChange={(e) => update("hidden", n(e.target.value))}
      />
      <ParameterField
        label="Attention heads"
        value={cfg.heads}
        placeholder="4"
        editable={editable}
        onChange={(e) => update("heads", n(e.target.value))}
      />
      <ParameterField
        label="GAT layers"
        value={cfg.num_layers}
        placeholder="2"
        editable={editable}
        onChange={(e) => update("num_layers", n(e.target.value))}
      />
      <ParameterField
        label="Dropout rate"
        value={cfg.dropout_rate}
        placeholder="0.1"
        editable={editable}
        min="0"
        max="1"
        step="0.05"
        onChange={(e) => update("dropout_rate", n(e.target.value))}
      />
      <ParameterField
        label="Learning rate"
        value={cfg.lr}
        placeholder="0.0004"
        editable={editable}
        min="0"
        step="0.0001"
        onChange={(e) => update("lr", n(e.target.value))}
      />
      <ParameterField
        label="Number of epochs"
        value={cfg.epochs}
        placeholder="5"
        editable={editable}
        onChange={(e) => update("epochs", n(e.target.value))}
      />
      <ParameterField
        label="Batch size"
        value={cfg.batch_size}
        placeholder="64"
        editable={editable}
        onChange={(e) => update("batch_size", n(e.target.value))}
      />
      <ParameterField
        label="Early stopping patience"
        value={cfg.patience}
        placeholder="10"
        editable={editable}
        onChange={(e) => update("patience", n(e.target.value))}
      />
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
  editable,
  min,
  max,
  step,
  helpText,
  onChange,
}: {
  label: string;
  value: number;
  placeholder: string;
  editable: boolean;
  min?: string;
  max?: string;
  step?: string;
  helpText?: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      {editable ? (
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          className="w-full bg-white text-black border rounded px-3 py-2 appearance-none focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
      ) : (
        <div className="px-3 py-2 font-medium text-black">{value}</div>
      )}
      {helpText ? <div className="mt-2 text-xs text-gray-500">{helpText}</div> : null}
    </div>
  );
}

function Icon({ selected }: { selected: boolean }) {
  return (
    <div className={`p-3 rounded-lg transition ${selected ? "bg-brand-500" : "bg-gray-100"}`}>
      <Settings className={`w-5 h-5 ${selected ? "text-white" : "text-gray-600"}`} />
    </div>
  );
}

function Radio({ selected }: { selected: boolean }) {
  return (
    <div
      className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
        selected ? "border-brand-500" : "border-gray-300"
      }`}
    >
      {selected && <div className="w-2.5 h-2.5 bg-brand-500 rounded-full" />}
    </div>
  );
}


