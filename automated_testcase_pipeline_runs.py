import json
import os
import time
import traceback
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from conv_and_viz.xes_to_csv import convert_xes_to_csv
from conv_and_viz.preprocessor_csv import preprocess_event_log
from ppm_pipeline import (
    default_transformer_config,
    default_gnn_config,
    run_next_activity_prediction,
    run_event_time_prediction,
    run_remaining_time_prediction,
    run_gnn_unified_prediction,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "testcase_dataset")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "testrun_results")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SPLIT = 0.5

# Write the test case numbers you want to run here.
# Example: [1, 2, 15, 30]
# Leave empty [] to run all 30 test cases.
SELECTED_TEST_CASES = [2, 3, 8, 9, 14, 15, 20, 21, 26, 27]

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _filter_selected_specs(specs, selected_cases):
    if not selected_cases:
        return specs

    selected_set = set(selected_cases)
    filtered_specs = [spec for idx, spec in enumerate(specs, 1) if idx in selected_set]
    invalid_cases = sorted(selected_set - set(range(1, len(specs) + 1)))
    return filtered_specs, invalid_cases


class TestcaseCombinationRunner:
    def __init__(self, dataset_dir, output_dir, test_size, val_split):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.val_split = val_split
        self.session_tag = _now_tag()
        self.session_dir = os.path.join(self.output_dir, f"pipeline_combo_{self.session_tag}")
        self.prep_dir = os.path.join(self.session_dir, "prepared_datasets")
        self.master_log = os.path.join(self.session_dir, "master_log.txt")
        self.results = []

        os.makedirs(self.prep_dir, exist_ok=True)
        self._write_master_header()

    def _write_master_header(self):
        with open(self.master_log, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("TESTCASE PIPELINE COMBINATION RUN\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 90 + "\n\n")

    def log(self, message):
        print(message)
        with open(self.master_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def list_xes_files(self):
        if not os.path.isdir(self.dataset_dir):
            self.log(f"[X] Dataset directory not found: {self.dataset_dir}")
            return []
        files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(".xes")]
        files.sort()
        self.log(f"Found {len(files)} XES dataset(s) in {self.dataset_dir}:")
        for i, file_name in enumerate(files, 1):
            self.log(f"  {i}. {file_name}")
        return files

    def _split_and_save(self, df, out_dir):
        required = ["CaseID", "Activity", "Timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for split generation: {missing}")

        work = df.copy()
        work["__split"] = ""

        trainval_idx, test_idx = train_test_split(
            work.index,
            test_size=self.test_size,
            random_state=42,
            shuffle=True,
        )
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=self.val_split,
            random_state=42,
            shuffle=True,
        )

        work.loc[train_idx, "__split"] = "train"
        work.loc[val_idx, "__split"] = "val"
        work.loc[test_idx, "__split"] = "test"

        split_dir = os.path.join(out_dir, "splits")
        os.makedirs(split_dir, exist_ok=True)

        work.loc[work["__split"] == "train"].to_csv(os.path.join(split_dir, "train.csv"), index=False)
        work.loc[work["__split"] == "val"].to_csv(os.path.join(split_dir, "val.csv"), index=False)
        work.loc[work["__split"] == "test"].to_csv(os.path.join(split_dir, "test.csv"), index=False)

        return work

    def _find_custom_target(self, df):
        ignore_cols = {"CaseID", "Activity", "Timestamp", "Resource", "__split"}
        for col in df.columns:
            if col in ignore_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].nunique(dropna=True) >= 2:
                return col
        return None

    def prepare_dataset(self, xes_file):
        base_name = os.path.splitext(xes_file)[0]
        dataset_root = os.path.join(self.prep_dir, base_name)
        os.makedirs(dataset_root, exist_ok=True)

        xes_path = os.path.join(self.dataset_dir, xes_file)

        self.log(f"\n[STEP 1] Preparing dataset: {xes_file}")

        csv_path, _, _ = convert_xes_to_csv(xes_path, dataset_root)

        preprocessed_path = os.path.join(dataset_root, f"{base_name}_preprocessed.csv")
        preprocess_event_log(csv_path, preprocessed_path)

        df = pd.read_csv(preprocessed_path)

        rename_map = {}
        for src, tgt in [
            ("case:concept:name", "CaseID"),
            ("case:id", "CaseID"),
            ("concept:name", "Activity"),
            ("time:timestamp", "Timestamp"),
            ("org:resource", "Resource"),
        ]:
            if src in df.columns and tgt not in df.columns:
                rename_map[src] = tgt

        if rename_map:
            df = df.rename(columns=rename_map)

        with_splits = self._split_and_save(df, dataset_root)
        split_dataset_path = os.path.join(dataset_root, f"{base_name}_with_splits.csv")
        with_splits.to_csv(split_dataset_path, index=False)

        custom_target = self._find_custom_target(with_splits)

        meta = {
            "source_xes": xes_path,
            "converted_csv": csv_path,
            "preprocessed_csv": preprocessed_path,
            "split_dataset": split_dataset_path,
            "split": {"test_size": self.test_size, "val_split": self.val_split},
            "custom_target": custom_target,
            "rows": int(len(with_splits)),
            "columns": list(with_splits.columns),
        }
        with open(os.path.join(dataset_root, "dataset_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.log(f"[OK] Prepared: {base_name} | rows={len(with_splits)} | custom_target={custom_target}")

        return {
            "name": base_name,
            "dataset_path": split_dataset_path,
            "custom_target": custom_target,
            "prepared_dir": dataset_root,
        }

    def build_specs(self, prepared):
        specs = []

        model_specs = {
            "transformer": {
                "config": default_transformer_config(),
                "explainability": "all",  # Both (LIME + SHAP)
                "tasks": ["next_activity", "event_time", "remaining_time"],
            },
            "gnn": {
                "config": default_gnn_config(),
                "explainability": "all",  # Both (Gradient + GraphLIME)
                "tasks": ["next_activity", "event_time", "remaining_time"],
            },
        }

        for dataset in prepared:
            for model_name, cfg in model_specs.items():
                for task in cfg["tasks"]:
                    specs.append({
                        "dataset_name": dataset["name"],
                        "dataset_path": dataset["dataset_path"],
                        "prepared_dir": dataset["prepared_dir"],
                        "model": model_name,
                        "task": task,
                        "config": cfg["config"],
                        "explainability": cfg["explainability"],
                        "custom_target": dataset["custom_target"],
                    })

        return specs

    def _run_transformer(self, spec, run_dir):
        if spec["task"] == "next_activity":
            return run_next_activity_prediction(
                spec["dataset_path"],
                run_dir,
                self.test_size,
                self.val_split,
                spec["config"],
                explainability_method=spec["explainability"],
                skip_auto_mapping=False,
            )
        if spec["task"] == "event_time":
            return run_event_time_prediction(
                spec["dataset_path"],
                run_dir,
                self.test_size,
                self.val_split,
                spec["config"],
                explainability_method=spec["explainability"],
                skip_auto_mapping=False,
            )
        if spec["task"] == "remaining_time":
            return run_remaining_time_prediction(
                spec["dataset_path"],
                run_dir,
                self.test_size,
                self.val_split,
                spec["config"],
                explainability_method=spec["explainability"],
                skip_auto_mapping=False,
            )
        raise RuntimeError(f"Unsupported transformer task: {spec['task']}")

    def _run_gnn(self, spec, run_dir):
        return run_gnn_unified_prediction(
            spec["dataset_path"],
            run_dir,
            self.test_size,
            self.val_split,
            spec["config"],
            explainability_method=spec["explainability"],
            task=spec["task"],
            target_column=None,
            skip_auto_mapping=False,
        )

    def run_specs(self, specs):
        total = len(specs)
        for i, spec in enumerate(specs, 1):
            run_name = f"run_{i:03d}_{spec['dataset_name']}_{spec['model']}_{spec['task']}"
            run_name = run_name.replace(" ", "_")
            run_dir = os.path.join(self.session_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            self.log("\n" + "-" * 90)
            self.log(f"[STEP 6] RUN {i}/{total}: {spec['dataset_name']} | {spec['model']} | {spec['task']}")
            self.log("-" * 90)

            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "steps": {
                        "1": "Select XES, run preprocessing, generate splits",
                        "2": "Use both model types (transformer + gnn)",
                        "3": "Use default model configs",
                        "4": "Use all 2 classification + all 2 regression tasks",
                        "5": "Use both explainability methods",
                        "6": "Execute full pipeline",
                    },
                    "dataset": spec["dataset_path"],
                    "dataset_name": spec["dataset_name"],
                    "model": spec["model"],
                    "task": spec["task"],
                    "custom_target": spec["custom_target"],
                    "split": {"test_size": self.test_size, "val_split": self.val_split},
                    "config": spec["config"],
                    "explainability": spec["explainability"],
                }, f, indent=2)

            row = {
                "run": i,
                "dataset": spec["dataset_name"],
                "model": spec["model"],
                "task": spec["task"],
                "custom_target": spec["custom_target"],
                "output_dir": run_dir,
                "success": False,
                "duration_s": None,
                "metrics": {},
                "error": None,
            }

            start = time.time()
            try:
                if spec["model"] == "transformer":
                    metrics = self._run_transformer(spec, run_dir)
                else:
                    metrics = self._run_gnn(spec, run_dir)

                row["success"] = True
                row["metrics"] = metrics or {}

                with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(row["metrics"], f, indent=2)

                self.log("[OK] Run completed successfully")

            except Exception as exc:
                row["error"] = str(exc)
                err_path = os.path.join(run_dir, "ERROR_LOG.txt")
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                self.log(f"[X] Run failed: {exc}")
                self.log(f"[X] Trace: {err_path}")

            row["duration_s"] = round(time.time() - start, 2)
            self.results.append(row)

    def save_summary(self):
        summary_path = os.path.join(self.session_dir, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        totals = {
            "total": len(self.results),
            "success": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
        }
        with open(os.path.join(self.session_dir, "summary_totals.json"), "w", encoding="utf-8") as f:
            json.dump(totals, f, indent=2)

        self.log("\n" + "=" * 90)
        self.log("TEST RUN COMPLETE")
        self.log(f"Results directory: {self.session_dir}")
        self.log(f"Total runs: {totals['total']} | Success: {totals['success']} | Failed: {totals['failed']}")
        self.log(f"Summary: {summary_path}")
        self.log("=" * 90)

    def run_all(self):
        xes_files = self.list_xes_files()
        if not xes_files:
            self.log("[X] No .xes files found. Exiting.")
            return

        prepared = []
        for xes_file in xes_files:
            try:
                prepared.append(self.prepare_dataset(xes_file))
            except Exception as exc:
                self.log(f"[X] Failed to prepare dataset {xes_file}: {exc}")

        if not prepared:
            self.log("[X] No datasets could be prepared. Exiting.")
            return

        self.log("\n[STEP 2-5] Building full model/task/explainability combinations...")
        specs = self.build_specs(prepared)
        self.log(f"Total planned runs: {len(specs)}")

        specs, invalid_cases = _filter_selected_specs(specs, SELECTED_TEST_CASES)
        if SELECTED_TEST_CASES:
            self.log(f"Selected test cases: {SELECTED_TEST_CASES}")
            self.log(f"Filtered planned runs: {len(specs)}")
        if invalid_cases:
            self.log(f"[WARNING] Ignoring invalid test case numbers: {invalid_cases}")
        if not specs:
            self.log("[X] No valid selected test cases to run. Exiting.")
            return

        self.run_specs(specs)
        self.save_summary()


def main():
    runner = TestcaseCombinationRunner(
        dataset_dir=DEFAULT_DATASET_DIR,
        output_dir=DEFAULT_RESULTS_DIR,
        test_size=DEFAULT_TEST_SIZE,
        val_split=DEFAULT_VAL_SPLIT,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
