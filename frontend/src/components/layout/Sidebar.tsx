type SidebarProps = {
  currentStep: number;
  completedSteps: boolean[];
};

const STEPS = [
  "Dataset Setup & Mapping",
  "Select Model Type",
  "Model Configuration",
  "Select Prediction Task",
  "Select Explainability Method",
  "Review & Run",
  "Results",
];

export default function Sidebar({
  currentStep,
  completedSteps,
}: SidebarProps) {
  return (
    <aside className="w-72 border-r border-brand-100 bg-white pt-20 px-6 pb-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-semibold text-brand-900">
          Pipeline Configuration
        </h1>
        <p className="text-sm text-brand-600">
          PPMExplainer
        </p>
      </div>

      {/* Steps */}
      <nav className="space-y-2">
        {STEPS.map((label, index) => {
          const isActive = index === currentStep;
          const isCompleted = completedSteps[index];

          return (
            <div
              key={label}
              className={`flex items-center gap-4 rounded-lg px-4 py-3 ${
                isActive
                  ? "bg-brand-50 border-l-4 border-brand-600"
                  : ""
              }`}
            >
              {/* Step Indicator */}
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
                  isCompleted
                    ? "border border-green-500 text-green-600 bg-green-50"
                    : isActive
                    ? "bg-brand-600 text-white"
                    : "border border-brand-200 text-brand-400"
                }`}
              >
                {isCompleted ? "✓" : index + 1}
              </div>

              {/* Label */}
              <span
                className={`text-sm ${
                  isActive
                    ? "font-medium text-brand-900"
                    : isCompleted
                    ? "text-brand-600"
                    : "text-brand-400"
                }`}
              >
                {label}
              </span>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}

