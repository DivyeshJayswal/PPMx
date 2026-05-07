type TopHeaderProps = {
  currentStep: number;
};

const STEP_TITLES = [
  "Dataset Setup & Mapping",
  "Select Model Type",
  "Model Configuration",
  "Prediction Task",
  "Explainability",
  "Review & Run",
  "Results",
];

export default function TopHeader({
  currentStep,
}: TopHeaderProps) {
  return (
    <div className="mb-6">
      <p className="text-sm text-gray-500">
        Step {currentStep + 1} of {STEP_TITLES.length}
      </p>
      <h1 className="text-2xl font-semibold">
        {STEP_TITLES[currentStep]}
      </h1>
    </div>
  );
}
