# backend/runner/run_job.py
import argparse
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def patch_status(status_path: str, **patch) -> None:
    status: Dict[str, Any] = {}
    if os.path.exists(status_path):
        status = read_json(status_path)
    status.update(patch)
    status["updated_at"] = utc_now()
    write_json(status_path, status)


def list_artifacts(artifacts_dir: str) -> list[str]:
    out: list[str] = []
    if not os.path.exists(artifacts_dir):
        return out
    for root, _, files in os.walk(artifacts_dir):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, artifacts_dir)
            out.append(rel)
    out.sort()
    return out


def artifact_bucket_name() -> str | None:
    return os.getenv("ARTIFACT_BUCKET") or os.getenv("UPLOAD_BUCKET")


def upload_run_dir_to_gcs(run_dir: str, run_id: str) -> None:
    bucket_name = artifact_bucket_name()
    if not bucket_name:
        return

    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError(f"google-cloud-storage is required for artifact upload: {e}") from e

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = f"runs/{run_id}/"

    for root, _, files in os.walk(run_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, run_dir).replace(os.sep, "/")
            bucket.blob(prefix + rel_path).upload_from_filename(full_path)


def normalize_explainability(value: Any) -> Any:
    # Frontend/Swagger often sends "none"
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def _apply_manual_column_mapping(
    dataset_path: str,
    out_csv_path: str,
    mapping: Dict[str, Any],
    target_column: str | None = None,
) -> str:
    """
    Create a mapped CSV with canonical columns:
      CaseID, Activity, Timestamp, (optional) Resource
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(f"Manual column mapping requires pandas. Import failed: {e}")

    required_keys = {"case_id", "activity", "timestamp"}
    missing = required_keys - set(mapping.keys())
    if missing:
        raise RuntimeError(f"Invalid column_mapping: missing keys {sorted(missing)}")

    case_col = str(mapping.get("case_id") or "").strip()
    act_col = str(mapping.get("activity") or "").strip()
    ts_col = str(mapping.get("timestamp") or "").strip()
    res_col = mapping.get("resource")
    res_col = str(res_col).strip() if isinstance(res_col, str) else ""

    if not case_col or not act_col or not ts_col:
        raise RuntimeError("Invalid column_mapping: case_id, activity, timestamp must be non-empty")

    selected = [case_col, act_col, ts_col] + ([res_col] if res_col else [])
    if len(set(selected)) != len(selected):
        raise RuntimeError("Invalid column_mapping: selected columns must be unique")

    df = pd.read_csv(dataset_path)
    for col in [case_col, act_col, ts_col] + ([res_col] if res_col else []):
        if col and col not in df.columns:
            raise RuntimeError(f"Invalid column_mapping: column not found: {col}")

    if target_column:
        if target_column not in df.columns:
            raise RuntimeError(f"Invalid target_column: column not found: {target_column}")

    rename = {case_col: "CaseID", act_col: "Activity", ts_col: "Timestamp"}
    if res_col:
        rename[res_col] = "Resource"

    df = df.rename(columns=rename)
    keep_cols = ["CaseID", "Activity", "Timestamp"] + (["Resource"] if res_col else [])
    if target_column and target_column not in keep_cols:
        keep_cols.append(target_column)
    if "__split" in df.columns:
        keep_cols.append("__split")
    df = df[keep_cols]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    request_path = os.path.join(run_dir, "request.json")
    status_path = os.path.join(run_dir, "status.json")
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    req = read_json(request_path)
    run_id = req["run_id"]
    dataset_id = req["dataset_id"]

    # Resolve dataset path from registry
    datasets_meta_path = os.path.abspath(
        os.path.join(run_dir, "..", "..", "datasets", dataset_id, "meta.json")
    )
    if not os.path.exists(datasets_meta_path):
        patch_status(
            status_path,
            status="failed",
            finished_at=utc_now(),
            error=f"Dataset meta not found: {datasets_meta_path}",
        )
        raise RuntimeError(f"Dataset meta not found: {datasets_meta_path}")

    dataset_meta = read_json(datasets_meta_path)
    dataset_path = dataset_meta["stored_path"]

    model_type = (req.get("model_type") or "").lower().strip()
    task = (req.get("task") or "").lower().strip()
    target_column = req.get("target_column") or None

    split = req.get("split") or {"test_size": 0.2, "val_split": 0.5}
    test_size = float(split.get("test_size", 0.2))
    val_split = float(split.get("val_split", 0.5))

    config = req.get("config") or {}
    explainability = normalize_explainability(req.get("explainability", None))

    mapping_mode = (req.get("mapping_mode") or "").strip().lower() or "auto"
    column_mapping = req.get("column_mapping") or None
    skip_auto_mapping = mapping_mode == "manual"
    if task == "custom_activity" and not target_column:
        raise RuntimeError("custom_activity requires target_column")

    if mapping_mode not in {"auto", "manual"}:
        patch_status(
            status_path,
            status="failed",
            finished_at=utc_now(),
            error=f"Invalid mapping_mode: {mapping_mode}",
        )
        raise RuntimeError(f"Invalid mapping_mode: {mapping_mode}")

    try:
        if mapping_mode == "manual":
            if not isinstance(column_mapping, dict):
                raise RuntimeError("mapping_mode=manual requires column_mapping")

            mapped_path = os.path.join(run_dir, "input", "dataset_mapped.csv")
            dataset_path = _apply_manual_column_mapping(
                dataset_path, mapped_path, column_mapping, target_column=target_column
            )
    except Exception as e:
        # If we fail before the normal "running" patch, ensure we don't leave the run stuck in queued.
        patch_status(status_path, status="failed", finished_at=utc_now(), error=str(e))
        raise

    # Mark running BEFORE any heavy imports
    patch_status(status_path, status="running", started_at=utc_now(), error=None)

    # Write early summary so artifacts are never empty while running
    write_json(os.path.join(artifacts_dir, "summary.json"), {
        "run_id": run_id,
        "status": "running",
        "dataset": {
            "dataset_id": dataset_id,
            "filename": os.path.basename(dataset_path),
            "num_events": dataset_meta.get("num_events"),
            "num_cases": dataset_meta.get("num_cases"),
        },
        "request": req,
        "artifacts": list_artifacts(artifacts_dir),
        "started_at": utc_now(),
    })

    try:
        # Lazy import (prevents "queued forever" on import failure)
        from ppm_pipeline import (
            default_gnn_config,
            default_transformer_config,
            run_next_activity_prediction,
            run_event_time_prediction,
            run_remaining_time_prediction,
            run_gnn_unified_prediction,
        )

        if model_type == "transformer" and not config:
            config = default_transformer_config()
        if model_type == "gnn" and not config:
            config = default_gnn_config()

        if model_type == "transformer":
            if task in {"next_activity", "custom_activity"}:
                metrics = run_next_activity_prediction(
                    dataset_path,
                    artifacts_dir,
                    test_size,
                    val_split,
                    config,
                    explainability_method=explainability,
                    target_column=target_column if task == "custom_activity" else None,
                    skip_auto_mapping=skip_auto_mapping,
                )
            elif task == "event_time":
                metrics = run_event_time_prediction(
                    dataset_path, artifacts_dir, test_size, val_split, config,
                    explainability_method=explainability,
                    skip_auto_mapping=skip_auto_mapping,
                )
            elif task == "remaining_time":
                metrics = run_remaining_time_prediction(
                    dataset_path, artifacts_dir, test_size, val_split, config,
                    explainability_method=explainability,
                    skip_auto_mapping=skip_auto_mapping,
                )
            else:
                raise RuntimeError(f"Unsupported transformer task: {task}")

        elif model_type == "gnn":
            if task not in {"next_activity", "custom_activity", "event_time", "remaining_time", "unified"}:
                raise RuntimeError(f"Unsupported gnn task: {task}")

            gnn_task = "next_activity" if task == "custom_activity" else task
            metrics = run_gnn_unified_prediction(
                dataset_path,
                artifacts_dir,
                test_size,
                val_split,
                config,
                explainability_method=explainability,
                task=gnn_task,
                target_column=target_column if task == "custom_activity" else None,
                skip_auto_mapping=skip_auto_mapping,
            )
        else:
            raise RuntimeError(f"Unsupported model_type: {model_type}")

        # metrics.json for frontend rendering
        write_json(os.path.join(artifacts_dir, "metrics.json"), {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "model_type": model_type,
            "task": task,
            "split": {"test_size": test_size, "val_split": val_split},
            "config": config,
            "metrics": metrics,
            "finished_at": utc_now(),
        })

        # final summary.json
        write_json(os.path.join(artifacts_dir, "summary.json"), {
            "run_id": run_id,
            "status": "succeeded",
            "dataset": {
                "dataset_id": dataset_id,
                "filename": os.path.basename(dataset_path),
                "num_events": dataset_meta.get("num_events"),
                "num_cases": dataset_meta.get("num_cases"),
            },
            "request": req,
            "metrics": metrics,
            "artifacts": list_artifacts(artifacts_dir),
            "finished_at": utc_now(),
        })

        patch_status(status_path, status="succeeded", finished_at=utc_now(), error=None)
        try:
            upload_run_dir_to_gcs(run_dir, run_id)
        except Exception as upload_error:
            patch_status(status_path, artifact_upload_error=str(upload_error))

    except Exception as e:
        tb = traceback.format_exc()
        write_json(os.path.join(artifacts_dir, "error.json"), {"error": str(e), "traceback": tb})
        write_json(os.path.join(artifacts_dir, "summary.json"), {
            "run_id": run_id,
            "status": "failed",
            "dataset": {
                "dataset_id": dataset_id,
                "filename": os.path.basename(dataset_path),
                "num_events": dataset_meta.get("num_events"),
                "num_cases": dataset_meta.get("num_cases"),
            },
            "request": req,
            "artifacts": list_artifacts(artifacts_dir),
            "error": str(e),
            "finished_at": utc_now(),
        })
        patch_status(status_path, status="failed", finished_at=utc_now(), error=str(e))
        try:
            upload_run_dir_to_gcs(run_dir, run_id)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
