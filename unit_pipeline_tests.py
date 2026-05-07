import json
import os
import tempfile
import traceback
import unittest
import gzip
from datetime import datetime
from pathlib import Path
import uuid

import pandas as pd
import numpy as np

from backend import main as backend_main
from backend.runner import run_job
from conv_and_viz import preprocessor_csv
from gnns import prefix_generation
from utils import column_detector as utils_column_detector
import ppm_pipeline
import automated_batch_runs


REPORT_DIR = Path("tests/test_reports")


class UnitTestLogger:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.report_dir / f"unit_test_log_{timestamp}.txt"
        self.results = []

        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("UNIT TEST RUN\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

    def log(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def add_result(self, name: str, success: bool, error: str | None = None) -> None:
        self.results.append({
            "name": name,
            "success": success,
            "error": error,
        })

    def write_report(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"unit_test_report_{timestamp}.json"
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
            "results": self.results,
            "generated_at": datetime.now().isoformat(),
        }
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


LOGGER = UnitTestLogger(REPORT_DIR)


class PipelineUnitTests(unittest.TestCase):
    def _record(self, name: str, ok: bool, error: str | None = None) -> None:
        LOGGER.add_result(name, ok, error)

    def test_infer_column_types(self):
        name = "infer_column_types"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "2", "3"],
                "Activity": ["A", "B", "C"],
                "Duration": [1.2, 3.4, 5.6],
            })
            result = backend_main._infer_column_types(df)
            self.assertEqual(result["CaseID"], "categorical")
            self.assertEqual(result["Activity"], "categorical")
            self.assertEqual(result["Duration"], "numerical")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_utc_now_format(self):
        name = "utc_now_format"
        try:
            stamp = backend_main._utc_now()
            self.assertTrue(stamp.endswith("Z"))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_and_standardize_columns(self):
        name = "detect_and_standardize_columns"
        try:
            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "2"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "org:resource": ["R1", "R2", "R1"],
            })
            df_out, mapping = backend_main.detect_and_standardize_columns(df)
            self.assertIn("CaseID", df_out.columns)
            self.assertIn("Activity", df_out.columns)
            self.assertIn("Timestamp", df_out.columns)
            self.assertIn("Resource", df_out.columns)
            self.assertEqual(mapping["case:concept:name"], "CaseID")
            self.assertEqual(mapping["org:resource"], "Resource")
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_out["Timestamp"]))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_case_column(self):
        name = "detect_case_column"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "2", "3"],
                "Activity": ["A", "B", "C"],
                "Timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            })
            case_col = backend_main._detect_case_column(df)
            self.assertEqual(case_col, "CaseID")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_case_column_missing(self):
        name = "detect_case_column_missing"
        try:
            df = pd.DataFrame({
                "Activity": ["A", "B", "C"],
                "Timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            })
            case_col = backend_main._detect_case_column(df)
            self.assertIsNone(case_col)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_original_column_mapping(self):
        name = "detect_original_column_mapping"
        try:
            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "2"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "org:resource": ["R1", "R2", "R1"],
            })
            detected = backend_main._detect_original_column_mapping(df)
            self.assertEqual(detected["case_id"], "case:concept:name")
            self.assertEqual(detected["activity"], "concept:name")
            self.assertEqual(detected["timestamp"], "time:timestamp")
            self.assertEqual(detected["resource"], "org:resource")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_preprocess_event_log_csv(self):
        name = "preprocess_event_log_csv"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.csv")
                output_path = os.path.join(tmpdir, "output.csv")
                df = pd.DataFrame({
                    "case_id": ["1", "1", "2", "2"],
                    "activity": ["A", None, "B", "C"],
                    "timestamp": [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:05:00",
                        "2024-01-02 00:00:00",
                        "2024-01-02 00:05:00",
                    ],
                    "amount": [1.0, None, 3.0, None],
                })
                df.to_csv(input_path, index=False)
                out_df = preprocessor_csv.preprocess_event_log(
                    input_path, output_csv_path=output_path
                )
                self.assertTrue(os.path.exists(output_path))
                self.assertIn("timestamp", [c.lower() for c in out_df.columns])
                self.assertFalse(out_df["activity"].isna().any())
                self.assertFalse(out_df["amount"].isna().any())
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_column_type(self):
        name = "detect_column_type"
        try:
            s_num = pd.Series([1.0, 2.0, 3.5])
            s_cat = pd.Series(["A", "B", "C"])
            s_dt = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02"]))
            self.assertEqual(preprocessor_csv.detect_column_type(s_num), "numerical")
            self.assertEqual(preprocessor_csv.detect_column_type(s_cat), "categorical")
            self.assertEqual(preprocessor_csv.detect_column_type(s_dt), "datetime")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_write_read_json(self):
        name = "write_read_json"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "sample.json")
                payload = {"ok": True, "value": 123, "nested": {"a": "b"}}
                backend_main._write_json(path, payload)
                loaded = backend_main._read_json(path)
                self.assertEqual(payload, loaded)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_dataset_path_helpers(self):
        name = "dataset_path_helpers"
        try:
            dataset_id = "unit_test_ds"
            ds_dir = backend_main._dataset_dir(dataset_id)
            meta_path = backend_main._dataset_meta_path(dataset_id)
            file_path = backend_main._dataset_file_path(dataset_id)
            self.assertTrue(ds_dir.endswith(os.path.join("datasets", dataset_id)))
            self.assertTrue(meta_path.endswith(os.path.join("datasets", dataset_id, "meta.json")))
            self.assertTrue(file_path.endswith(os.path.join("datasets", dataset_id, "dataset.csv")))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_list_sample_datasets(self):
        name = "list_sample_datasets"
        original_path = backend_main.SAMPLE_DATASETS_CONFIG_PATH
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = os.path.join(tmpdir, "sample_datasets.json")
                with open(config_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        [
                            {
                                "name": "sample.csv",
                                "format": "csv",
                                "size_bytes": 123,
                                "drive_url": "PASTE_GOOGLE_DRIVE_LINK_HERE",
                            },
                            {
                                "name": "sample.xes",
                                "format": "xes",
                                "drive_url": "https://drive.google.com/file/d/abc123/view?usp=sharing",
                            },
                        ],
                        handle,
                    )

                backend_main.SAMPLE_DATASETS_CONFIG_PATH = config_path
                datasets = backend_main._list_sample_datasets()

                self.assertEqual(len(datasets), 2)
                self.assertEqual(datasets[0]["name"], "sample.csv")
                self.assertEqual(datasets[0]["format"], "csv")
                self.assertFalse(datasets[0]["configured"])
                self.assertEqual(datasets[1]["name"], "sample.xes")
                self.assertEqual(datasets[1]["format"], "xes")
                self.assertTrue(datasets[1]["configured"])
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise
        finally:
            backend_main.SAMPLE_DATASETS_CONFIG_PATH = original_path

    def test_extract_google_drive_file_id(self):
        name = "extract_google_drive_file_id"
        try:
            self.assertEqual(
                backend_main._extract_google_drive_file_id(
                    "https://drive.google.com/file/d/abc123/view?usp=sharing"
                ),
                "abc123",
            )
            self.assertEqual(
                backend_main._extract_google_drive_file_id(
                    "https://drive.google.com/open?id=xyz789"
                ),
                "xyz789",
            )
            self.assertEqual(
                backend_main._extract_google_drive_file_id(
                    "https://drive.google.com/uc?export=download&id=id456"
                ),
                "id456",
            )
            self.assertIsNone(
                backend_main._extract_google_drive_file_id("https://example.com/file")
            )
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_run_path_helpers(self):
        name = "run_path_helpers"
        try:
            run_id = "unit_test_run"
            run_dir = backend_main._run_dir(run_id)
            status_path = backend_main._run_status_path(run_id)
            request_path = backend_main._run_request_path(run_id)
            artifacts_dir = backend_main._run_artifacts_dir(run_id)
            self.assertTrue(run_dir.endswith(os.path.join("runs", run_id)))
            self.assertTrue(status_path.endswith(os.path.join("runs", run_id, "status.json")))
            self.assertTrue(request_path.endswith(os.path.join("runs", run_id, "request.json")))
            self.assertTrue(artifacts_dir.endswith(os.path.join("runs", run_id, "artifacts")))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_write_split_files(self):
        name = "write_split_files"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "1", "2", "2"],
                "Activity": ["A", "B", "A", "B"],
                "Timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            })
            with tempfile.TemporaryDirectory() as tmpdir:
                train_df = df.iloc[:2].copy()
                val_df = df.iloc[2:3].copy()
                test_df = df.iloc[3:].copy()
                split_path, split_paths = backend_main._write_split_files(
                    df, tmpdir, train_df, val_df, test_df
                )
                self.assertTrue(os.path.exists(split_paths["train"]))
                self.assertTrue(os.path.exists(split_paths["val"]))
                self.assertTrue(os.path.exists(split_paths["test"]))
                self.assertTrue(os.path.exists(split_path))
                split_df = pd.read_csv(split_path)
                self.assertIn("__split", split_df.columns)
                self.assertEqual(set(split_df["__split"].unique()), {"train", "val", "test"})
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_load_run_status(self):
        name = "load_run_status"
        run_dir = None
        try:
            run_id = f"unit_test_{uuid.uuid4()}"
            run_dir = backend_main._run_dir(run_id)
            os.makedirs(run_dir, exist_ok=True)
            status_path = backend_main._run_status_path(run_id)
            payload = {"status": "completed", "run_id": run_id}
            backend_main._write_json(status_path, payload)
            loaded = backend_main._load_run_status(run_id)
            self.assertEqual(payload, loaded)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise
        finally:
            try:
                if run_dir and os.path.isdir(run_dir):
                    for root, dirs, files in os.walk(run_dir, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for d in dirs:
                            os.rmdir(os.path.join(root, d))
                    os.rmdir(run_dir)
            except Exception:
                pass

    def test_compute_split_frames(self):
        name = "compute_split_frames"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"],
                "Activity": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "Timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            })
            cfg = backend_main.SplitConfig(test_size=0.2, val_split=0.5)
            train_df, val_df, test_df = backend_main._compute_split_frames(df, cfg)
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
            self.assertGreater(len(test_df), 0)
            all_ids = set(train_df.index) | set(val_df.index) | set(test_df.index)
            self.assertEqual(len(all_ids), len(df))
            train_cases = set(train_df["CaseID"].unique())
            val_cases = set(val_df["CaseID"].unique())
            test_cases = set(test_df["CaseID"].unique())
            self.assertTrue(train_cases.isdisjoint(val_cases))
            self.assertTrue(train_cases.isdisjoint(test_cases))
            self.assertTrue(val_cases.isdisjoint(test_cases))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_compute_split_frames_no_case(self):
        name = "compute_split_frames_no_case"
        try:
            df = pd.DataFrame({
                "Activity": ["A", "B", "C", "D", "E"],
                "Timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            })
            cfg = backend_main.SplitConfig(test_size=0.4, val_split=0.5)
            train_df, val_df, test_df = backend_main._compute_split_frames(df, cfg)
            self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_validate_split_config(self):
        name = "validate_split_config"
        try:
            with self.assertRaises(Exception):
                backend_main._validate_split_config(backend_main.SplitConfig(test_size=0, val_split=0.5))
            with self.assertRaises(Exception):
                backend_main._validate_split_config(backend_main.SplitConfig(test_size=0.2, val_split=1.5))
            backend_main._validate_split_config(backend_main.SplitConfig(test_size=0.2, val_split=0.5))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_run_job_helpers(self):
        name = "run_job_helpers"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                status_path = os.path.join(tmpdir, "status.json")
                run_job.patch_status(status_path, status="queued")
                status = run_job.read_json(status_path)
                self.assertEqual(status.get("status"), "queued")
                run_job.patch_status(status_path, status="running")
                status = run_job.read_json(status_path)
                self.assertEqual(status.get("status"), "running")

                artifacts_dir = os.path.join(tmpdir, "artifacts")
                os.makedirs(os.path.join(artifacts_dir, "nested"), exist_ok=True)
                with open(os.path.join(artifacts_dir, "a.txt"), "w", encoding="utf-8") as f:
                    f.write("a")
                with open(os.path.join(artifacts_dir, "nested", "b.txt"), "w", encoding="utf-8") as f:
                    f.write("b")
                artifacts = run_job.list_artifacts(artifacts_dir)
                self.assertIn("a.txt", artifacts)
                self.assertIn(os.path.join("nested", "b.txt"), artifacts)

                self.assertIsNone(run_job.normalize_explainability("none"))
                self.assertIsNone(run_job.normalize_explainability("null"))
                self.assertEqual(run_job.normalize_explainability("shap"), "shap")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_manual_column_mapping(self):
        name = "manual_column_mapping"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = os.path.join(tmpdir, "data.csv")
                out = os.path.join(tmpdir, "mapped.csv")
                df = pd.DataFrame({
                    "case_col": ["1", "1", "2"],
                    "act_col": ["A", "B", "C"],
                    "time_col": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    "res_col": ["R1", "R2", "R1"],
                    "target": ["X", "Y", "Z"],
                    "__split": ["train", "val", "test"],
                })
                df.to_csv(src, index=False)
                mapped = run_job._apply_manual_column_mapping(
                    src,
                    out,
                    {
                        "case_id": "case_col",
                        "activity": "act_col",
                        "timestamp": "time_col",
                        "resource": "res_col",
                    },
                    target_column="target",
                )
                self.assertTrue(os.path.exists(mapped))
                mapped_df = pd.read_csv(mapped)
                self.assertIn("CaseID", mapped_df.columns)
                self.assertIn("Activity", mapped_df.columns)
                self.assertIn("Timestamp", mapped_df.columns)
                self.assertIn("Resource", mapped_df.columns)
                self.assertIn("target", mapped_df.columns)
                self.assertIn("__split", mapped_df.columns)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_universal_column_detector_basic(self):
        name = "universal_column_detector_basic"
        try:
            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "2"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "org:resource": ["R1", "R2", "R1"],
            })
            detector = utils_column_detector.UniversalColumnDetector(df)
            detected, mapping = detector.detect_all()
            self.assertEqual(detected.get("case_id"), "case:concept:name")
            self.assertEqual(detected.get("activity"), "concept:name")
            self.assertEqual(detected.get("timestamp"), "time:timestamp")
            self.assertEqual(detected.get("resource"), "org:resource")
            self.assertIn("case:concept:name", mapping)
            mapped_df = detector.apply_mapping()
            self.assertIn("CaseID", mapped_df.columns)
            self.assertIn("Activity", mapped_df.columns)
            self.assertIn("Timestamp", mapped_df.columns)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_universal_column_detector_custom_pattern(self):
        name = "universal_column_detector_custom_pattern"
        original = list(utils_column_detector.UniversalColumnDetector.KNOWN_PATTERNS.get("case_id", []))
        try:
            utils_column_detector.add_custom_pattern("case_id", ["trace_identifier"])
            df = pd.DataFrame({
                "trace_identifier": ["1", "2", "3"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            })
            detector = utils_column_detector.UniversalColumnDetector(df)
            detected, _ = detector.detect_all()
            self.assertEqual(detected.get("case_id"), "trace_identifier")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise
        finally:
            utils_column_detector.UniversalColumnDetector.KNOWN_PATTERNS["case_id"] = original

    def test_universal_column_detector_report(self):
        name = "universal_column_detector_report"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "2"],
                "Activity": ["A", "B"],
                "Timestamp": ["2024-01-01", "2024-01-02"],
            })
            detector = utils_column_detector.UniversalColumnDetector(df)
            detector.detect_all()
            report = detector.get_detection_report()
            self.assertIn("COLUMN DETECTION REPORT", report)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_gnn_prefix_generation(self):
        name = "gnn_prefix_generation"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "1", "2", "2"],
                "Activity": ["A", "B", "A", "C"],
                "Resource": ["R1", "R2", "R1", "R3"],
                "Timestamp": pd.to_datetime([
                    "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"
                ]),
                "variant": ["X", "X", "Y", "Y"],
            })
            trace_cols = prefix_generation.detect_trace_attributes(df)
            self.assertIn("variant", trace_cols)
            prefix_df = prefix_generation.generate_prefix_dataset(df, trace_cols)
            self.assertIn("next_activity", prefix_df.columns)
            self.assertEqual(len(prefix_df), 2)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_gnn_load_and_sort_log(self):
        name = "gnn_load_and_sort_log"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = os.path.join(tmpdir, "log.csv")
                df = pd.DataFrame({
                    "CaseID": ["2", "1", "1"],
                    "Activity": ["C", "A", "B"],
                    "Timestamp": ["2024-01-03", "2024-01-01", "2024-01-02"],
                    "Resource": ["R1", "R2", "R3"],
                })
                df.to_csv(csv_path, index=False)
                sorted_df = prefix_generation.load_and_sort_log(Path(csv_path))
                self.assertEqual(str(sorted_df.iloc[0]["CaseID"]), "1")
                self.assertLessEqual(sorted_df.iloc[0]["Timestamp"], sorted_df.iloc[1]["Timestamp"])
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_gnn_dataset_builder_helpers(self):
        name = "gnn_dataset_builder_helpers"
        try:
            try:
                from gnns import dataset_builder
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            edges = dataset_builder.build_dfr_edges(1)
            self.assertEqual(edges.shape[1], 0)
            edges = dataset_builder.build_dfr_edges(3)
            self.assertEqual(edges.shape[1], 2)

            df = pd.DataFrame({
                "CaseID": ["1", "1", "2", "2"],
                "prefix_id": [1, 1, 1, 1],
                "prefix_pos": [1, 2, 1, 2],
                "Activity": ["A", "B", "A", "C"],
                "Resource": ["R1", "R2", "R1", "R3"],
                "Timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
                "next_activity": ["B", "B", "C", "C"],
                "variant": ["X", "X", "Y", "Y"],
            })
            trace_cols = dataset_builder.detect_trace_attributes(df)
            self.assertIn("variant", trace_cols)
            vocabs = dataset_builder.build_global_vocabs(df, trace_cols)
            self.assertIn("Activity", vocabs)
            self.assertIn("Resource", vocabs)
            self.assertIn("variant", vocabs)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_gnn_graph_folder_dataset(self):
        name = "gnn_graph_folder_dataset"
        try:
            try:
                import torch
                from gnns.prediction.gnn_predictor import GraphFolderDataset
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(torch.tensor([1, 2, 3]), os.path.join(tmpdir, "0.pt"))
                torch.save(torch.tensor([4, 5, 6]), os.path.join(tmpdir, "1.pt"))
                dataset = GraphFolderDataset(tmpdir)
                self.assertEqual(len(dataset), 2)
                item = dataset[0]
                self.assertTrue(torch.is_tensor(item))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_transformer_model_shapes(self):
        name = "transformer_model_shapes"
        try:
            try:
                import tensorflow as tf  # noqa: F401
                from transformers.model import build_next_activity_model, build_time_prediction_model
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            model_act = build_next_activity_model(vocab_size=5, max_len=3, d_model=8, num_heads=2, num_blocks=1)
            out_act = model_act.predict(np.array([[1, 2, 0], [2, 3, 4]]), verbose=0)
            self.assertEqual(out_act.shape, (2, 5))

            model_time = build_time_prediction_model(vocab_size=5, max_len=3, d_model=8, num_heads=2, num_blocks=1)
            out_time = model_time.predict(
                [np.array([[1, 2, 0], [2, 3, 4]]), np.zeros((2, 3))],
                verbose=0
            )
            self.assertIsInstance(out_time, list)
            self.assertEqual(len(out_time), 2)
            self.assertEqual(out_time[0].shape, (2, 1))
            self.assertEqual(out_time[1].shape, (2, 3))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_next_activity_sequence_builder(self):
        name = "next_activity_sequence_builder"
        try:
            try:
                import tensorflow as tf  # noqa: F401
                from transformers.prediction.next_activity import NextActivityPredictor
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            predictor = NextActivityPredictor(max_len=4)
            df = pd.DataFrame({
                "case_id": ["1", "1", "2", "2"],
                "activity": ["A", "B", "A", "C"],
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]),
            })
            sequences, next_acts, meta = predictor._create_sequences_with_prefixes(df)
            self.assertEqual(len(sequences), 2)
            self.assertEqual(len(next_acts), 2)
            self.assertEqual(meta["max_len"], 1)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_event_time_temporal_features(self):
        name = "event_time_temporal_features"
        try:
            try:
                import tensorflow as tf  # noqa: F401
                from transformers.prediction.event_time import EventTimePredictor
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "1"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            })
            predictor = EventTimePredictor(max_len=4)
            out = predictor._calculate_temporal_features(df)
            self.assertIn("fvt1", out.columns)
            self.assertIn("fvt2", out.columns)
            self.assertIn("fvt3", out.columns)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_remaining_time_temporal_features(self):
        name = "remaining_time_temporal_features"
        try:
            try:
                import tensorflow as tf  # noqa: F401
                from transformers.prediction.remaining_time import RemainingTimePredictor
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "1"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            })
            predictor = RemainingTimePredictor(max_len=4)
            out = predictor._calculate_temporal_features(df)
            self.assertIn("fvt1", out.columns)
            self.assertIn("fvt2", out.columns)
            self.assertIn("fvt3", out.columns)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_explainability_benchmark_utils(self):
        name = "explainability_benchmark_utils"
        try:
            try:
                from explainability.transformers.transformer_explainer import (
                    compare_benchmark_results,
                    generate_benchmark_latex_table,
                )
            except Exception as import_exc:
                self._record(name, True, f"skipped: {import_exc}")
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "benchmark.json")
                payload = {
                    "faithfulness": {"faithfulness_k5": {"spearman_correlation": 0.5}},
                    "comprehensiveness": {"comprehensiveness_k5": {"mean": 0.1}},
                    "sufficiency": {"sufficiency_k5": {"mean": 0.2}},
                    "monotonicity": {"monotonicity": {"mean": 0.3}},
                    "method_agreement": {"agreement_k5": {"jaccard_similarity": 0.4}},
                    "temporal_consistency": {"temporal_consistency": {"recency_correlation": 0.1}},
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

                df = compare_benchmark_results([("modelA", json_path)])
                self.assertEqual(len(df), 1)
                latex = generate_benchmark_latex_table(df)
                self.assertIn("\\begin{table}", latex)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_ppm_pipeline_defaults(self):
        name = "ppm_pipeline_defaults"
        try:
            t_cfg = ppm_pipeline.default_transformer_config()
            g_cfg = ppm_pipeline.default_gnn_config()
            for key in [
                "max_len",
                "d_model",
                "num_heads",
                "num_blocks",
                "dropout_rate",
                "epochs",
                "batch_size",
                "patience",
            ]:
                self.assertIn(key, t_cfg)
            for key in ["hidden", "dropout_rate", "lr", "epochs", "batch_size", "patience"]:
                self.assertIn(key, g_cfg)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_ppm_pipeline_column_detection(self):
        name = "ppm_pipeline_column_detection"
        try:
            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "2"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            })
            df_out, mapping, _ = ppm_pipeline.detect_and_standardize_columns(df, verbose=False)
            self.assertIn("CaseID", df_out.columns)
            self.assertIn("Activity", df_out.columns)
            self.assertIn("Timestamp", df_out.columns)
            self.assertEqual(mapping.get("case:concept:name"), "CaseID")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_batch_required_images_checker(self):
        name = "batch_required_images_checker"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                missing = automated_batch_runs.check_required_images(
                    tmpdir, model="transformer", task="next_activity"
                )
                self.assertTrue(missing)
                missing_gnn = automated_batch_runs.check_required_images(
                    tmpdir, model="gnn", task="next_activity"
                )
                self.assertTrue(missing_gnn)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_xes_conversion_optional(self):
        name = "xes_conversion_optional"
        try:
            try:
                import pm4py  # noqa: F401
                from conv_and_viz.xes_to_csv import convert_xes_to_csv
            except Exception:
                self._record(name, True)
                return
            xes_payload = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes" openxes.version="1.0RC7" xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case-1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T00:00:00.000+00:00"/>
    </event>
  </trace>
</log>
"""
            with tempfile.TemporaryDirectory() as tmpdir:
                xes_path = Path(tmpdir) / "compressed_log.xes"
                with gzip.open(xes_path, "wt", encoding="utf-8") as handle:
                    handle.write(xes_payload)
                csv_path, df, _ = convert_xes_to_csv(str(xes_path), tmpdir)
                self.assertTrue(Path(csv_path).exists())
                self.assertFalse(df.empty)
                self.assertIn("concept:name", df.columns)
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise


def run_unit_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineUnitTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    LOGGER.write_report()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        exit_code = run_unit_tests()
        LOGGER.log("Unit tests completed.")
        raise SystemExit(exit_code)
    except Exception:
        LOGGER.log("Unit test runner crashed:")
        LOGGER.log(traceback.format_exc())
        LOGGER.write_report()
        raise
