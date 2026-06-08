type SidebarProps = {
  currentStep: number;
  completedSteps: boolean[];
  accessibleSteps: boolean[];
};

const STEPS = [
  "Dataset Setup & Mapping",
  "Select Model Type",
  "Model Configuration",
  "Select Prediction Task",
  "Select Explainability Method",
  "Explainability Configuration",
  "Review & Run",
  "Results",
];

export default function Sidebar({
  currentStep,
  completedSteps,
  accessibleSteps,
}: SidebarProps) {
  return (
    <aside className="w-72 border-r border-brand-100 bg-white pt-20 px-6 pb-6">
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-brand-900">PPMExplainer</h1>
      </div>

      <nav className="space-y-2">
        {STEPS.map((label, index) => {
          const isActive = index === currentStep;
          const isCompleted = completedSteps[index];
          const isAccessible = accessibleSteps[index];
          const isLocked = !isAccessible && !isActive;

          return (
            <div
              key={label}
              className={`flex items-center gap-4 rounded-lg px-4 py-3 ${
                isActive
                  ? "bg-brand-50 border-l-4 border-brand-600"
                  : isLocked
                  ? "opacity-60"
                  : ""
              }`}
              aria-disabled={isLocked}
            >
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
                  isCompleted
                    ? "border border-green-500 text-green-600 bg-green-50"
                    : isActive
                    ? "bg-brand-600 text-white"
                    : isLocked
                    ? "border border-gray-200 text-gray-300 bg-gray-50"
                    : "border border-brand-200 text-brand-400"
                }`}
              >
                {isCompleted ? "✓" : isLocked ? "•" : index + 1}
              </div>

              <span
                className={`text-sm ${
                  isActive
                    ? "font-medium text-brand-900"
                    : isCompleted
                    ? "text-brand-600"
                    : isLocked
                    ? "text-gray-400"
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
