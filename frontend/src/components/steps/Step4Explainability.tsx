// frontend/src/components/steps/Step4Explainability.tsx
import Card from "../ui/card";

// NOTE: Values must match backend expectations.
// - transformer: "lime" | "shap"
// - gnn: "gradient" | "lime" (GraphLIME)
// - both: "all"
// - none: "none" (runner normalizes to null)
export type ExplainValue = "lime" | "shap" | "gradient" | "all" | "none";

type Step4ExplainabilityProps = {
  modelType: "gnn" | "transformer" | null;
  method: ExplainValue | null;
  onSelect: (method: ExplainValue) => void;
};

type Option = {
  value: ExplainValue;
  title: string;
  description: string;
};

export default function Step4Explainability({
  modelType,
  method,
  onSelect,
}: Step4ExplainabilityProps) {
  const options: Option[] = !modelType
    ? []
    : modelType === "transformer"
    ? [
        {
          value: "none",
          title: "None",
          description: "Skip explainability to run faster.",
        },
        {
          value: "lime",
          title: "LIME",
          description:
            "Local surrogate explanations. Explains individual predictions by approximating the model locally with an interpretable model.",
        },
        {
          value: "shap",
          title: "SHAP",
          description:
            "Shapley-value based feature attributions. Provides consistent local explanations across features.",
        },
        {
          value: "all",
          title: "Both (LIME + SHAP)",
          description: "Run both methods (takes longer).",
        },
      ]
    : [
        {
          value: "none",
          title: "None",
          description: "Skip explainability to run faster.",
        },
        {
          value: "gradient",
          title: "Gradient-Based",
          description:
            "Uses gradients to estimate which input features influence predictions most strongly.",
        },
        {
          value: "lime",
          title: "GraphLIME",
          description:
            "Graph-specific local explanations. Identifies important substructures/features for a prediction.",
        },
        {
          value: "all",
          title: "Both (Gradient + GraphLIME)",
          description: "Run both methods (takes longer).",
        },
      ];

  return (
    <div className="space-y-8 w-full">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Select Explainability Method</h2>
        <p className="text-sm text-brand-600">
          Methods are filtered automatically based on the selected model type.
        </p>
      </div>

      {!modelType ? (
        <Card>
          <div className="p-6 text-sm text-gray-700">
            Please select a model type in Step 2 first.
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {options.map((opt) => {
            const selected = method === opt.value;

            return (
              <div
                key={opt.value}
                className={[
                  "border rounded-xl p-6 bg-white cursor-pointer transition",
                  selected ? "border-brand-500 ring-1 ring-brand-200" : "hover:border-gray-300",
                ].join(" ")}
                onClick={() => onSelect(opt.value)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onSelect(opt.value);
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-lg font-semibold">{opt.title}</div>
                    <div className="text-sm text-gray-600 mt-1">{opt.description}</div>
                  </div>

                  <div
                    className={[
                      "h-5 w-5 rounded-full border flex items-center justify-center mt-1",
                      selected ? "border-brand-600" : "border-gray-300",
                    ].join(" ")}
                    aria-hidden="true"
                  >
                    {selected && <div className="h-3 w-3 rounded-full bg-brand-600" />}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Info */}
      <Card title="Note">
        <div className="bg-brand-50 border border-brand-200 rounded-lg p-4 text-sm text-gray-700">
          <ul className="list-disc ml-5 space-y-1">
            <li>Transformer models: LIME, SHAP, both, or none</li>
            <li>GNN models: Gradient-Based, GraphLIME, both, or none</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
