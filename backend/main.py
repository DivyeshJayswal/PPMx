# backend/main.py
import os
import sys
import uuid
import json
import shutil
import subprocess
import tempfile
import zipfile
from urllib.parse import parse_qs, urlparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import threading

import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from collections import deque
import re

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="PPM Backend", version="0.1.0")

# Allow the Vite frontend to call the API during development
app.add_middleware(
    CORSMiddleware,
    # Allow localhost dev servers on any port (Vite commonly uses 5173/5174, etc.)
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _strip_api_prefix(request: Request, call_next):
    """
    Firebase Hosting forwards /api/* to Cloud Run unchanged.
    Our backend routes are defined without /api, so strip it here.
    """
    path = request.scope.get("path", "")
    if path == "/api" or path.startswith("/api/"):
        new_path = path[4:] or "/"
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode("utf-8")
    return await call_next(request)

# -----------------------------------------------------------------------------
# Paths / Storage
# -----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))            # .../repo/backend
REPO_ROOT = os.path.dirname(BACKEND_DIR)                            # .../repo
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # allow importing project-level modules

# Import preprocessing utilities
try:
    from conv_and_viz.preprocessor_csv import preprocess_event_log
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("[WARNING] Preprocessor not available - skipping data cleaning")

def _default_storage_dir() -> str:
    configured = os.getenv("PPM_STORAGE_DIR")
    if configured:
        return os.path.abspath(configured)

    if os.getenv("K_SERVICE"):
        return os.path.join(tempfile.gettempdir(), "ppm-backend-storage")

    return os.path.join(BACKEND_DIR, "storage")


STORAGE_DIR = _default_storage_dir()
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
RUNS_DIR = os.path.join(STORAGE_DIR, "runs")
SAMPLE_DATASETS_CONFIG_PATH = os.path.abspath(
    os.getenv("SAMPLE_DATASETS_CONFIG_PATH", os.path.join(BACKEND_DIR, "sample_datasets.json"))
)

for d in (STORAGE_DIR, UPLOAD_DIR, DATASETS_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "400"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_SAMPLE_DOWNLOAD_MB = int(os.getenv("MAX_SAMPLE_DOWNLOAD_MB", "400"))
MAX_SAMPLE_DOWNLOAD_BYTES = MAX_SAMPLE_DOWNLOAD_MB * 1024 * 1024

# Use the same Python interpreter that runs uvicorn (your backend/.venv)
PYTHON_EXEC = sys.executable

# P2.2: Process tracking for subprocess timeout management
# Maps run_id -> {"proc": Popen, "started_at": datetime, "pid": int}
_active_processes: Dict[str, Dict] = {}
RUN_TIMEOUT_SECONDS = 7200  # 2 hours max per run

# P2.3: Storage management
RUN_TTL_HOURS = 72         # Auto-cleanup runs older than 72h
MAX_STORAGE_MB = 10000     # 10GB disk quota for runs

# -----------------------------------------------------------------------------
# Small JSON helpers (atomic write)
# -----------------------------------------------------------------------------
def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_sample_dataset_catalog() -> List[Dict[str, Any]]:
    if not os.path.isfile(SAMPLE_DATASETS_CONFIG_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Sample dataset catalog not found: {SAMPLE_DATASETS_CONFIG_PATH}",
        )
    data = _read_json(SAMPLE_DATASETS_CONFIG_PATH)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Invalid sample dataset catalog format.")
    return [item for item in data if isinstance(item, dict)]


def _sample_dataset_catalog_status() -> Dict[str, Any]:
    exists = os.path.isfile(SAMPLE_DATASETS_CONFIG_PATH)
    status: Dict[str, Any] = {
        "path": SAMPLE_DATASETS_CONFIG_PATH,
        "exists": exists,
        "count": 0,
        "error": None,
    }
    if not exists:
        status["error"] = "sample_datasets.json is missing from the running container"
        return status
    try:
        data = _read_json(SAMPLE_DATASETS_CONFIG_PATH)
        if not isinstance(data, list):
            status["error"] = "catalog JSON root is not a list"
            return status
        status["count"] = len([item for item in data if isinstance(item, dict)])
    except Exception as e:
        status["error"] = str(e)
    return status


def _is_drive_url_configured(url: Optional[str]) -> bool:
    if not url:
        return False
    stripped = url.strip()
    if not stripped:
        return False
    if "PASTE_GOOGLE_DRIVE_LINK_HERE" in stripped:
        return False
    return True


def _get_sample_dataset_entry(name: str) -> Dict[str, Any]:
    for item in _load_sample_dataset_catalog():
        if item.get("name") == name:
            return item
    raise HTTPException(status_code=404, detail="Sample dataset not found.")


def _list_sample_datasets() -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for item in sorted(_load_sample_dataset_catalog(), key=lambda entry: str(entry.get("name", "")).lower()):
        name = str(item.get("name", "")).strip()
        fmt = str(item.get("format", "")).strip().lower()
        if not name or fmt not in {"csv", "xes"}:
            continue
        size_bytes = item.get("size_bytes")
        datasets.append(
            {
                "name": name,
                "size_bytes": int(size_bytes) if isinstance(size_bytes, (int, float)) else None,
                "format": fmt,
                "configured": _is_drive_url_configured(item.get("drive_url")),
            }
        )
    return datasets


def _extract_google_drive_file_id(link: str) -> Optional[str]:
    parsed = urlparse(link)
    if "drive.google.com" not in parsed.netloc.lower():
        return None

    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    parts = [part for part in parsed.path.split("/") if part]
    if "d" in parts:
        idx = parts.index("d")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _google_drive_download_response(drive_url: str) -> requests.Response:
    file_id = _extract_google_drive_file_id(drive_url)
    if not file_id:
        raise HTTPException(status_code=500, detail="Invalid Google Drive link in sample dataset catalog.")

    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(base_url, params=params, stream=True, timeout=120)

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        response.close()
        response = session.get(
            base_url,
            params={"export": "download", "id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=120,
        )

    if response.status_code >= 400:
        response.close()
        raise HTTPException(status_code=502, detail="Failed to download sample dataset from Google Drive.")

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        body = response.text[:4000]
        response.close()
        raise HTTPException(
            status_code=502,
            detail="Google Drive returned an HTML confirmation page instead of the dataset file.",
        )

    return response


def _detect_upload_format(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xes") or lower.endswith(".xes.gz"):
        return "xes"
    return ""


# -----------------------------------------------------------------------------
# Column detection / standardization
# -----------------------------------------------------------------------------
def detect_and_standardize_columns(
    df: pd.DataFrame, verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detect typical event-log columns and standardize them to:
      - CaseID
      - Activity
      - Timestamp
      - Resource (optional)

    Returns:
      (df_renamed, mapping_old_to_new)
    """
    column_mapping: Dict[str, str] = {}

    case_patterns = [
        "case:id", "case:concept:name", "CaseID", "case_id", "caseid", "Case ID", "Case_ID"
    ]
    activity_patterns = [
        "concept:name", "Action", "activity", "event", "Event", "task", "Task", "Activity"
    ]
    timestamp_patterns = [
        "time:timestamp", "Timestamp", "timestamp", "time", "Time", "start_time",
        "StartTime", "complete_time", "CompleteTime"
    ]
    resource_patterns = [
        "org:resource", "Resource", "resource", "user", "User", "org:role",
        "role", "Role", "actor", "Actor"
    ]

    # Case
    for col in df.columns:
        if col in case_patterns and col != "CaseID":
            column_mapping[col] = "CaseID"
            break
    if "CaseID" in df.columns and "CaseID" not in column_mapping.values():
        # already ok
        pass

    # Activity
    for col in df.columns:
        if col in activity_patterns and col != "Activity":
            column_mapping[col] = "Activity"
            break

    # Timestamp
    for col in df.columns:
        if col in timestamp_patterns and col != "Timestamp":
            column_mapping[col] = "Timestamp"
            break

    # Resource (optional)
    for col in df.columns:
        if col in resource_patterns and col != "Resource":
            column_mapping[col] = "Resource"
            break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ["CaseID", "Activity", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")

    # Make timestamp parseable (do not fail hard; just best effort)
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    except Exception:
        pass

    if verbose:
        print("COLUMN DETECTION:", column_mapping)

    return df, column_mapping


def _detect_original_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect likely event-log columns without renaming them.

    Returns keys expected by the frontend mapping UI:
      - case_id
      - activity
      - timestamp
      - resource (optional)
    """
    detected: Dict[str, str] = {}

    case_patterns = [
        "case:id", "case:concept:name", "CaseID", "case_id", "caseid", "Case ID", "Case_ID"
    ]
    activity_patterns = [
        "concept:name", "Action", "activity", "event", "Event", "task", "Task", "Activity"
    ]
    timestamp_patterns = [
        "time:timestamp", "Timestamp", "timestamp", "time", "Time", "start_time",
        "StartTime", "complete_time", "CompleteTime"
    ]
    resource_patterns = [
        "org:resource", "Resource", "resource", "user", "User", "org:role",
        "role", "Role", "actor", "Actor"
    ]

    for col in df.columns:
        if col in case_patterns:
            detected["case_id"] = col
            break

    for col in df.columns:
        if col in activity_patterns:
            detected["activity"] = col
            break

    for col in df.columns:
        if col in timestamp_patterns:
            detected["timestamp"] = col
            break

    for col in df.columns:
        if col in resource_patterns:
            detected["resource"] = col
            break

    return detected


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class DatasetUploadResponse(BaseModel):
    dataset_id: str
    stored_path: str
    raw_path: Optional[str] = None
    preprocessed_path: Optional[str] = None
    split_dataset_path: Optional[str] = None
    split_paths: Optional[Dict[str, str]] = None
    split_source: Optional[str] = None  # generated | uploaded
    split_config: Optional[Dict[str, float]] = None
    is_preprocessed: bool = False
    preprocessed_at: Optional[str] = None
    num_events: int = 0
    num_cases: int = 0
    columns: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    detected_mapping: Dict[str, str] = Field(default_factory=dict)
    column_diagnostics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    conversion_status: Optional[str] = None  # "converting" | "ready" | "failed"
    conversion_error: Optional[str] = None


class SampleDatasetInfo(BaseModel):
    name: str
    size_bytes: Optional[int] = None
    format: str
    configured: bool = False


class DatasetMeta(BaseModel):
    dataset_id: str
    stored_path: str
    raw_path: Optional[str] = None
    preprocessed_path: Optional[str] = None
    split_dataset_path: Optional[str] = None
    split_paths: Optional[Dict[str, str]] = None
    split_source: Optional[str] = None
    split_config: Optional[Dict[str, float]] = None
    is_preprocessed: bool = False
    preprocessed_at: Optional[str] = None
    preprocessing_options: Optional[Dict[str, bool]] = None
    num_events: int = 0
    num_cases: int = 0
    columns: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    detected_mapping: Dict[str, str] = Field(default_factory=dict)
    column_diagnostics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_at: str
    conversion_status: Optional[str] = None  # "converting" | "ready" | "failed"
    conversion_error: Optional[str] = None


class ColumnMapping(BaseModel):
    # Column names in the uploaded dataset
    case_id: str
    activity: str
    timestamp: str
    resource: Optional[str] = None


class ModelType(str, Enum):
    TRANSFORMER = "transformer"
    GNN = "gnn"


class TaskType(str, Enum):
    NEXT_ACTIVITY = "next_activity"
    CUSTOM_ACTIVITY = "custom_activity"
    EVENT_TIME = "event_time"
    REMAINING_TIME = "remaining_time"
    UNIFIED = "unified"


class ExplainabilityMethod(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    GRADIENT = "gradient"
    GRAPH_LIME = "graphlime"
    ALL = "all"


class RunCreateRequest(BaseModel):
    dataset_id: str
    model_type: ModelType = Field(..., description="transformer | gnn")
    task: TaskType = Field(..., description="next_activity | custom_activity | event_time | remaining_time | unified (gnn)")
    config: Dict[str, Any] = Field(default_factory=dict)
    split: Dict[str, float] = Field(default_factory=lambda: {"test_size": 0.2, "val_split": 0.5})
    explainability: Optional[ExplainabilityMethod] = None
    target_column: Optional[str] = None
    mapping_mode: Optional[str] = Field(
        default=None, description="auto | manual (optional; defaults to auto)"
    )
    column_mapping: Optional[ColumnMapping] = None

    @field_validator("task")
    @classmethod
    def validate_task_for_model(cls, v, info):
        """Validate task is compatible with model_type."""
        if 'model_type' in info.data:
            model_type = info.data['model_type']
            if model_type == ModelType.TRANSFORMER:
                allowed_tasks = {TaskType.NEXT_ACTIVITY, TaskType.CUSTOM_ACTIVITY, TaskType.EVENT_TIME, TaskType.REMAINING_TIME}
                if v not in allowed_tasks:
                    raise ValueError(f"Task '{v}' not supported for transformer model. Allowed: {[t.value for t in allowed_tasks]}")
            elif model_type == ModelType.GNN:
                allowed_tasks = {TaskType.NEXT_ACTIVITY, TaskType.EVENT_TIME, TaskType.REMAINING_TIME, TaskType.UNIFIED}
                if v not in allowed_tasks:
                    raise ValueError(f"Task '{v}' not supported for GNN model. Allowed: {[t.value for t in allowed_tasks]}")
        return v

    @field_validator("explainability")
    @classmethod
    def validate_explainability_for_model(cls, v, info):
        """Validate explainability method is compatible with model_type."""
        if v is None:
            return v
        if 'model_type' in info.data:
            model_type = info.data['model_type']
            if model_type == ModelType.TRANSFORMER:
                allowed_methods = {ExplainabilityMethod.SHAP, ExplainabilityMethod.LIME, ExplainabilityMethod.GRADIENT, ExplainabilityMethod.ALL}
                if v not in allowed_methods:
                    raise ValueError(f"Explainability '{v}' not supported for transformer. Allowed: {[m.value for m in allowed_methods]}")
            elif model_type == ModelType.GNN:
                allowed_methods = {ExplainabilityMethod.GRAPH_LIME, ExplainabilityMethod.GRADIENT, ExplainabilityMethod.ALL}
                if v not in allowed_methods:
                    raise ValueError(f"Explainability '{v}' not supported for GNN. Allowed: {[m.value for m in allowed_methods]}")
        return v


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None


class PreprocessOptions(BaseModel):
    sort_and_normalize_timestamps: bool = True
    check_millisecond_order: bool = True
    impute_categorical: bool = True
    impute_numeric_neighbors: bool = True
    drop_cases_with_missing_timestamps: bool = False
    fill_remaining_missing: bool = True
    remove_duplicates: bool = True


class SplitConfig(BaseModel):
    test_size: float = 0.2
    val_split: float = 0.5


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def _dataset_dir(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_id)


def _dataset_meta_path(dataset_id: str) -> str:
    return os.path.join(_dataset_dir(dataset_id), "meta.json")


def _dataset_file_path(dataset_id: str) -> str:
    return os.path.join(_dataset_dir(dataset_id), "dataset.csv")


def _load_dataset_meta(dataset_id: str) -> DatasetMeta:
    meta_path = _dataset_meta_path(dataset_id)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetMeta(**_read_json(meta_path))


def _detect_case_column(df: pd.DataFrame) -> Optional[str]:
    case_patterns = ["case:id", "case:concept:name", "CaseID", "case_id", "caseid", "Case ID", "Case_ID"]
    for col in df.columns:
        if col in case_patterns:
            return col
    return None


def _infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = "numerical"
        else:
            out[col] = "categorical"
    return out


def _parse_datetime_diagnostic(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _compute_column_diagnostics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    diagnostics: Dict[str, Dict[str, Any]] = {}
    timestamp_name_hints = ("timestamp", "time", "date")

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True)) if non_null > 0 else 0
        unique_ratio = float(unique_count / non_null) if non_null > 0 else 0.0
        mean_group_size = float(non_null / unique_count) if unique_count > 0 else 0.0

        value_counts = series.value_counts(dropna=True)
        max_frequency = int(value_counts.iloc[0]) if not value_counts.empty else 0
        max_frequency_share = float(max_frequency / non_null) if non_null > 0 else 0.0

        timestamp_parse_ratio = 0.0
        if (
            pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or any(hint in str(col).lower() for hint in timestamp_name_hints)
        ):
            parsed = _parse_datetime_diagnostic(series)
            timestamp_parse_ratio = float(parsed.notna().sum() / non_null) if non_null > 0 else 0.0

        diagnostics[col] = {
            "non_null_count": non_null,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "mean_group_size": mean_group_size,
            "max_frequency_share": max_frequency_share,
            "timestamp_parse_ratio": timestamp_parse_ratio,
            "looks_event_unique": bool(non_null >= 20 and unique_ratio >= 0.98),
            "looks_timestamp_like": bool(
                timestamp_parse_ratio >= 0.8
                or any(hint in str(col).lower() for hint in timestamp_name_hints)
            ),
        }

    return diagnostics


def _write_split_files(
    df: pd.DataFrame,
    ds_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[str, Dict[str, str]]:
    splits_dir = os.path.join(ds_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_df = pd.concat(
        [
            train_df.assign(__split="train"),
            val_df.assign(__split="val"),
            test_df.assign(__split="test"),
        ],
        ignore_index=True,
    )
    split_dataset_path = os.path.join(ds_dir, "dataset_with_splits.csv")
    split_df.to_csv(split_dataset_path, index=False)

    return split_dataset_path, {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }


# -----------------------------------------------------------------------------
# Run helpers
# -----------------------------------------------------------------------------
def _run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def _run_status_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "status.json")


def _run_request_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "request.json")


def _run_artifacts_dir(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "artifacts")


def _load_run_status(run_id: str) -> Dict[str, Any]:
    status_path = _run_status_path(run_id)
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail="Run not found")
    return _read_json(status_path)
 
 
# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "ppm-backend"}


@app.get("/version")
def version():
    def _module_version(name: str) -> Dict[str, Optional[str]]:
        try:
            mod = __import__(name)
            return {"present": True, "version": getattr(mod, "__version__", None)}
        except Exception:
            return {"present": False, "version": None}

    return {
        "service": "ppm-backend",
        "revision": os.getenv("K_REVISION"),
        "configuration": os.getenv("K_CONFIGURATION"),
        "build_sha": os.getenv("BUILD_SHA"),
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_sample_download_mb": MAX_SAMPLE_DOWNLOAD_MB,
        "docs_url": "/api/docs",
        "runtime": {
            "storage_dir": STORAGE_DIR,
            "storage_dir_exists": os.path.isdir(STORAGE_DIR),
            "storage_dir_writable": os.access(STORAGE_DIR, os.W_OK),
            "sample_datasets": _sample_dataset_catalog_status(),
        },
        "deps": {
            "torch": _module_version("torch"),
            "torch_geometric": _module_version("torch_geometric"),
            "tensorflow": _module_version("tensorflow"),
            "sklearn": _module_version("sklearn"),
            "shap": _module_version("shap"),
            "lime": _module_version("lime"),
        },
    }


def _convert_xes_background(dataset_id: str, raw_path: str, stored_path: str, ds_dir: str):
    """Background thread: convert XES→CSV and update dataset meta."""
    try:
        from conv_and_viz.xes_to_csv import convert_xes_to_csv
        csv_path, df, _ = convert_xes_to_csv(raw_path, ds_dir)

        # Normalize to dataset.csv
        if os.path.abspath(csv_path) != os.path.abspath(stored_path):
            shutil.copyfile(csv_path, stored_path)

        df = pd.read_csv(stored_path)
        num_events = int(len(df))
        detected_mapping = _detect_original_column_mapping(df)
        case_col = detected_mapping.get("case_id")
        num_cases = int(df[case_col].nunique()) if case_col else 0
        preview_rows = df.head(20).to_dict(orient="records")
        column_types = _infer_column_types(df)
        column_diagnostics = _compute_column_diagnostics(df)

        # Load existing meta and update
        meta_path = _dataset_meta_path(dataset_id)
        existing = json.loads(open(meta_path, "r").read())
        existing.update({
            "stored_path": stored_path,
            "num_events": num_events,
            "num_cases": num_cases,
            "columns": list(df.columns),
            "column_types": column_types,
            "detected_mapping": detected_mapping,
            "column_diagnostics": column_diagnostics,
            "conversion_status": "ready",
            "conversion_error": None,
        })
        # Write preview to a separate file (too large for meta in some cases)
        _write_json(os.path.join(ds_dir, "preview.json"), preview_rows)
        _write_json(meta_path, existing)

    except Exception as e:
        meta_path = _dataset_meta_path(dataset_id)
        try:
            existing = json.loads(open(meta_path, "r").read())
            existing["conversion_status"] = "failed"
            existing["conversion_error"] = str(e)
            _write_json(meta_path, existing)
        except Exception:
            pass


def _finalize_saved_dataset(
    *,
    dataset_id: str,
    ext: str,
    stored_path: str,
    raw_path: str,
    ds_dir: str,
    preprocessed: bool,
) -> DatasetUploadResponse:
    if ext == "xes":
        try:
            from conv_and_viz.xes_to_csv import convert_xes_to_csv  # noqa: F401
        except ImportError:
            shutil.rmtree(ds_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail="XES support requires pm4py; install backend dependencies.",
            )

        meta = DatasetMeta(
            dataset_id=dataset_id,
            stored_path=stored_path,
            raw_path=raw_path,
            created_at=_utc_now(),
            conversion_status="converting",
        )
        _write_json(_dataset_meta_path(dataset_id), meta.model_dump())

        t = threading.Thread(
            target=_convert_xes_background,
            args=(dataset_id, raw_path, stored_path, ds_dir),
            daemon=True,
        )
        t.start()

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            stored_path=stored_path,
            raw_path=raw_path,
            conversion_status="converting",
        )

    try:
        df = pd.read_csv(stored_path)
    except Exception as e:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")

    num_events = int(len(df))
    detected_mapping = _detect_original_column_mapping(df)
    case_col = detected_mapping.get("case_id")
    num_cases = int(df[case_col].nunique()) if case_col else 0

    preview_rows = df.head(20).to_dict(orient="records")
    column_types = _infer_column_types(df)
    column_diagnostics = _compute_column_diagnostics(df)

    preprocessed_at = _utc_now() if preprocessed else None

    meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=stored_path,
        raw_path=raw_path,
        preprocessed_path=stored_path if preprocessed else None,
        split_dataset_path=None,
        split_paths=None,
        split_source=None,
        split_config=None,
        is_preprocessed=preprocessed,
        preprocessed_at=preprocessed_at,
        preprocessing_options=None,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        column_types=column_types,
        detected_mapping=detected_mapping,
        column_diagnostics=column_diagnostics,
        created_at=_utc_now(),
        conversion_status="ready",
    )
    _write_json(_dataset_meta_path(dataset_id), meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=stored_path,
        raw_path=raw_path,
        preprocessed_path=meta.preprocessed_path,
        split_dataset_path=None,
        split_paths=None,
        split_source=None,
        split_config=None,
        is_preprocessed=meta.is_preprocessed,
        preprocessed_at=preprocessed_at,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        column_types=column_types,
        detected_mapping=detected_mapping,
        column_diagnostics=column_diagnostics,
        preview=preview_rows,
        conversion_status="ready",
    )


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...), preprocessed: bool = False):
    """
    Upload a CSV or XES dataset. CSV is parsed synchronously.
    XES files are saved to disk and converted in a background thread.
    Poll GET /datasets/{dataset_id} until conversion_status == "ready".
    """
    filename = file.filename or "dataset.csv"
    ext = _detect_upload_format(filename)
    if ext not in {"csv", "xes"}:
        raise HTTPException(status_code=400, detail="Only CSV, XES, or XES.GZ files are supported.")
    if preprocessed and ext != "csv":
        raise HTTPException(status_code=400, detail="Preprocessed uploads must be CSV.")

    dataset_id = str(uuid.uuid4())
    ds_dir = _dataset_dir(dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    stored_path = _dataset_file_path(dataset_id)  # final normalized CSV path
    raw_path = stored_path if ext == "csv" else os.path.join(ds_dir, "dataset.xes")

    # Save stream to disk with size enforcement
    size = 0
    try:
        with open(raw_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
                    )
                out.write(chunk)
    finally:
        await file.close()

    if size == 0:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return _finalize_saved_dataset(
        dataset_id=dataset_id,
        ext=ext,
        stored_path=stored_path,
        raw_path=raw_path,
        ds_dir=ds_dir,
        preprocessed=preprocessed,
    )


@app.get("/sample-datasets", response_model=List[SampleDatasetInfo])
def list_sample_datasets():
    return _list_sample_datasets()


@app.get("/sample-datasets/{sample_name}")
def download_sample_dataset(sample_name: str):
    item = _get_sample_dataset_entry(sample_name)
    drive_url = item.get("drive_url")
    if not _is_drive_url_configured(drive_url):
        raise HTTPException(status_code=400, detail="Google Drive link not configured for this sample dataset.")

    upstream = _google_drive_download_response(str(drive_url))
    media_type = "text/csv" if str(item.get("format", "")).lower() == "csv" else "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{sample_name}"'}

    def iter_stream():
        try:
            for chunk in upstream.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(iter_stream(), media_type=media_type, headers=headers)


@app.post("/datasets/sample/{sample_name}", response_model=DatasetUploadResponse)
def import_sample_dataset(sample_name: str):
    item = _get_sample_dataset_entry(sample_name)
    fmt = str(item.get("format", "")).strip().lower()
    if fmt not in {"csv", "xes"}:
        raise HTTPException(status_code=500, detail="Invalid sample dataset format.")

    drive_url = item.get("drive_url")
    if not _is_drive_url_configured(drive_url):
        raise HTTPException(status_code=400, detail="Google Drive link not configured for this sample dataset.")

    dataset_id = str(uuid.uuid4())
    ds_dir = _dataset_dir(dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    stored_path = _dataset_file_path(dataset_id)
    raw_path = stored_path if fmt == "csv" else os.path.join(ds_dir, "dataset.xes")

    upstream = _google_drive_download_response(str(drive_url))
    size = 0
    try:
        with open(raw_path, "wb") as out:
            for chunk in upstream.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_SAMPLE_DOWNLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Sample dataset too large. Max allowed is {MAX_SAMPLE_DOWNLOAD_MB} MB.",
                    )
                out.write(chunk)
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise
    finally:
        upstream.close()

    if size == 0:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail="Downloaded sample dataset is empty.")

    return _finalize_saved_dataset(
        dataset_id=dataset_id,
        ext=fmt,
        stored_path=stored_path,
        raw_path=raw_path,
        ds_dir=ds_dir,
        preprocessed=False,
    )


@app.post("/datasets/{dataset_id}/preprocess", response_model=DatasetUploadResponse)
def preprocess_dataset(dataset_id: str, options: PreprocessOptions):
    """
    Run preprocessing on the raw dataset with user-selected options.
    """
    if not PREPROCESSOR_AVAILABLE:
        raise HTTPException(status_code=500, detail="Preprocessor not available on server.")

    meta = _load_dataset_meta(dataset_id)
    raw_path = meta.raw_path or meta.stored_path
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Raw dataset file not found.")

    ds_dir = _dataset_dir(dataset_id)
    preprocessed_path = meta.preprocessed_path or os.path.join(ds_dir, "dataset_preprocessed.csv")

    try:
        print(f"[Preprocessing] Cleaning dataset: {raw_path}")
        df = preprocess_event_log(raw_path, preprocessed_path, options.model_dump())
        print(f"[Preprocessing] Complete. Events: {len(df):,}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    num_events = int(len(df))

    detected_mapping = _detect_original_column_mapping(df)
    case_col = detected_mapping.get("case_id")
    num_cases = int(df[case_col].nunique()) if case_col else 0

    preview_rows = df.head(20).to_dict(orient="records")
    column_types = _infer_column_types(df)
    column_diagnostics = _compute_column_diagnostics(df)
    processed_at = _utc_now()

    updated_meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=preprocessed_path,
        raw_path=raw_path,
        preprocessed_path=preprocessed_path,
        split_dataset_path=meta.split_dataset_path,
        split_paths=meta.split_paths,
        split_source=meta.split_source,
        split_config=meta.split_config,
        is_preprocessed=True,
        preprocessed_at=processed_at,
        preprocessing_options=options.model_dump(),
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        column_types=column_types,
        detected_mapping=detected_mapping or meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        created_at=meta.created_at,
    )
    _write_json(_dataset_meta_path(dataset_id), updated_meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=preprocessed_path,
        raw_path=raw_path,
        preprocessed_path=preprocessed_path,
        split_dataset_path=meta.split_dataset_path,
        split_paths=meta.split_paths,
        split_source=meta.split_source,
        split_config=meta.split_config,
        is_preprocessed=True,
        preprocessed_at=processed_at,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        column_types=column_types,
        detected_mapping=detected_mapping or meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        preview=preview_rows,
    )


def _validate_split_config(cfg: SplitConfig) -> None:
    if cfg.test_size <= 0 or cfg.test_size >= 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1.")
    if cfg.val_split <= 0 or cfg.val_split >= 1:
        raise HTTPException(status_code=400, detail="val_split must be between 0 and 1.")


def _compute_split_frames(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_col = _detect_case_column(df)
    rng = np.random.RandomState(42)

    if case_col:
        cases = df[case_col].dropna().unique()
        rng.shuffle(cases)
        n_cases = len(cases)
        n_test = max(1, int(n_cases * cfg.test_size)) if n_cases > 0 else 0
        n_remaining = max(0, n_cases - n_test)
        n_val = max(1, int(n_remaining * cfg.val_split)) if n_remaining > 0 else 0

        test_cases = set(cases[:n_test])
        val_cases = set(cases[n_test:n_test + n_val])
        train_cases = set(cases[n_test + n_val:])

        train_df = df[df[case_col].isin(train_cases)].copy()
        val_df = df[df[case_col].isin(val_cases)].copy()
        test_df = df[df[case_col].isin(test_cases)].copy()
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_total = len(idx)
        n_test = int(n_total * cfg.test_size)
        n_remaining = max(0, n_total - n_test)
        n_val = int(n_remaining * cfg.val_split)

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

    # P1.2: Validate splits are non-empty
    if train_df.empty:
        raise ValueError("Train split is empty. Dataset may be too small or split configuration invalid.")
    if val_df.empty:
        raise ValueError("Validation split is empty. Dataset may be too small or split configuration invalid.")
    if test_df.empty:
        raise ValueError("Test split is empty. Dataset may be too small or split configuration invalid.")

    return train_df, val_df, test_df


async def _save_upload_csv(file: UploadFile, out_path: str) -> pd.DataFrame:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for splits.")

    size = 0
    try:
        with open(out_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
                    )
                out.write(chunk)
    finally:
        await file.close()

    try:
        return pd.read_csv(out_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")


@app.post("/datasets/{dataset_id}/splits/generate", response_model=DatasetUploadResponse)
def generate_splits(dataset_id: str, cfg: SplitConfig):
    _validate_split_config(cfg)

    meta = _load_dataset_meta(dataset_id)
    source_path = meta.preprocessed_path or meta.raw_path or meta.stored_path
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    df = pd.read_csv(source_path)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty; cannot generate splits.")

    try:
        train_df, val_df, test_df = _compute_split_frames(df, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Split generation failed: {str(e)}")
    split_dataset_path, split_paths = _write_split_files(df, _dataset_dir(dataset_id), train_df, val_df, test_df)

    num_events = int(len(df))
    case_col = _detect_case_column(df)
    num_cases = int(df[case_col].nunique()) if case_col else 0

    split_df = pd.read_csv(split_dataset_path)
    preview_rows = split_df.head(20).to_dict(orient="records")
    column_types = _infer_column_types(split_df)
    column_diagnostics = _compute_column_diagnostics(split_df)

    updated_meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=meta.raw_path,
        preprocessed_path=meta.preprocessed_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="generated",
        split_config={"test_size": cfg.test_size, "val_split": cfg.val_split},
        is_preprocessed=meta.is_preprocessed,
        preprocessed_at=meta.preprocessed_at,
        preprocessing_options=meta.preprocessing_options,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(split_df.columns),
        column_types=column_types,
        detected_mapping=meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        created_at=meta.created_at,
    )
    _write_json(_dataset_meta_path(dataset_id), updated_meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=meta.raw_path,
        preprocessed_path=meta.preprocessed_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="generated",
        split_config={"test_size": cfg.test_size, "val_split": cfg.val_split},
        is_preprocessed=meta.is_preprocessed,
        preprocessed_at=meta.preprocessed_at,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(split_df.columns),
        column_types=column_types,
        detected_mapping=meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        preview=preview_rows,
    )


@app.post("/datasets/{dataset_id}/splits/upload", response_model=DatasetUploadResponse)
async def upload_splits(dataset_id: str, train: UploadFile = File(...), val: UploadFile = File(...), test: UploadFile = File(...)):
    meta = _load_dataset_meta(dataset_id)
    ds_dir = _dataset_dir(dataset_id)
    splits_dir = os.path.join(ds_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df = await _save_upload_csv(train, train_path)
    val_df = await _save_upload_csv(val, val_path)
    test_df = await _save_upload_csv(test, test_path)

    if list(train_df.columns) != list(val_df.columns) or list(train_df.columns) != list(test_df.columns):
        raise HTTPException(status_code=400, detail="Train/val/test columns must match.")

    split_dataset_path, split_paths = _write_split_files(train_df, ds_dir, train_df, val_df, test_df)

    combined_df = pd.read_csv(split_dataset_path)
    num_events = int(len(combined_df))
    case_col = _detect_case_column(combined_df)
    num_cases = int(combined_df[case_col].nunique()) if case_col else 0

    preview_rows = combined_df.head(20).to_dict(orient="records")
    column_types = _infer_column_types(combined_df)
    column_diagnostics = _compute_column_diagnostics(combined_df)

    updated_meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=meta.raw_path,
        preprocessed_path=meta.preprocessed_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="uploaded",
        split_config=meta.split_config,
        is_preprocessed=meta.is_preprocessed,
        preprocessed_at=meta.preprocessed_at,
        preprocessing_options=meta.preprocessing_options,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(combined_df.columns),
        column_types=column_types,
        detected_mapping=meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        created_at=meta.created_at,
    )
    _write_json(_dataset_meta_path(dataset_id), updated_meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=meta.raw_path,
        preprocessed_path=meta.preprocessed_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="uploaded",
        split_config=meta.split_config,
        is_preprocessed=meta.is_preprocessed,
        preprocessed_at=meta.preprocessed_at,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(combined_df.columns),
        column_types=column_types,
        detected_mapping=meta.detected_mapping,
        column_diagnostics=column_diagnostics,
        preview=preview_rows,
    )


@app.post("/datasets/splits/upload", response_model=DatasetUploadResponse)
async def upload_splits_new_dataset(train: UploadFile = File(...), val: UploadFile = File(...), test: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    ds_dir = _dataset_dir(dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    splits_dir = os.path.join(ds_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df = await _save_upload_csv(train, train_path)
    val_df = await _save_upload_csv(val, val_path)
    test_df = await _save_upload_csv(test, test_path)

    if list(train_df.columns) != list(val_df.columns) or list(train_df.columns) != list(test_df.columns):
        raise HTTPException(status_code=400, detail="Train/val/test columns must match.")

    split_dataset_path, split_paths = _write_split_files(train_df, ds_dir, train_df, val_df, test_df)

    combined_df = pd.read_csv(split_dataset_path)
    num_events = int(len(combined_df))
    case_col = _detect_case_column(combined_df)
    num_cases = int(combined_df[case_col].nunique()) if case_col else 0

    preview_rows = combined_df.head(20).to_dict(orient="records")
    column_types = _infer_column_types(combined_df)
    column_diagnostics = _compute_column_diagnostics(combined_df)
    created_at = _utc_now()

    meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=None,
        preprocessed_path=split_dataset_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="uploaded",
        split_config=None,
        is_preprocessed=True,
        preprocessed_at=created_at,
        preprocessing_options=None,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(combined_df.columns),
        column_types=column_types,
        detected_mapping={},
        column_diagnostics=column_diagnostics,
        created_at=created_at,
    )
    _write_json(_dataset_meta_path(dataset_id), meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=split_dataset_path,
        raw_path=None,
        preprocessed_path=split_dataset_path,
        split_dataset_path=split_dataset_path,
        split_paths=split_paths,
        split_source="uploaded",
        split_config=None,
        is_preprocessed=True,
        preprocessed_at=created_at,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(combined_df.columns),
        column_types=column_types,
        detected_mapping={},
        column_diagnostics=column_diagnostics,
        preview=preview_rows,
    )


@app.get("/datasets/{dataset_id}/splits/{split_name}")
def download_split(dataset_id: str, split_name: str):
    if split_name not in {"train", "val", "test"}:
        raise HTTPException(status_code=400, detail="Invalid split name.")

    meta = _load_dataset_meta(dataset_id)
    if not meta.split_paths or split_name not in meta.split_paths:
        raise HTTPException(status_code=404, detail="Split not available.")

    path = meta.split_paths[split_name]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Split file not found.")

    return FileResponse(path, filename=f"{split_name}.csv")


@app.get("/datasets/{dataset_id}/preprocessed")
def download_preprocessed_dataset(dataset_id: str):
    """
    Download the preprocessed dataset CSV (if available).
    """
    meta = _load_dataset_meta(dataset_id)
    if not meta.preprocessed_path or not os.path.exists(meta.preprocessed_path):
        raise HTTPException(status_code=404, detail="Preprocessed dataset not available.")

    return FileResponse(meta.preprocessed_path, filename="dataset_preprocessed.csv")


@app.get("/datasets/{dataset_id}", response_model=DatasetMeta)
def get_dataset(dataset_id: str):
    """
    Fetch dataset metadata from the registry.
    """
    return _load_dataset_meta(dataset_id)


def _get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def _cleanup_old_runs():
    """P2.3: Remove completed runs older than TTL_HOURS to free disk space."""
    if not os.path.isdir(RUNS_DIR):
        return 0

    cutoff = datetime.utcnow()
    removed = 0

    for run_id in os.listdir(RUNS_DIR):
        run_dir = os.path.join(RUNS_DIR, run_id)
        if not os.path.isdir(run_dir):
            continue

        status_path = os.path.join(run_dir, "status.json")
        if not os.path.exists(status_path):
            continue

        try:
            status = json.loads(open(status_path, "r").read())
        except Exception:
            continue

        # Only clean completed/failed runs
        if status.get("status") not in ("succeeded", "failed"):
            continue

        finished_at = status.get("finished_at")
        if not finished_at:
            continue

        try:
            finished_dt = datetime.fromisoformat(finished_at.rstrip("Z"))
        except Exception:
            continue

        age_hours = (cutoff - finished_dt).total_seconds() / 3600
        if age_hours > RUN_TTL_HOURS:
            try:
                shutil.rmtree(run_dir)
                removed += 1
            except Exception:
                pass

    return removed


def _enforce_disk_quota():
    """P2.3: If runs exceed disk quota, remove oldest completed runs."""
    if not os.path.isdir(RUNS_DIR):
        return

    current_mb = _get_dir_size_mb(RUNS_DIR)
    if current_mb <= MAX_STORAGE_MB:
        return

    # Collect completed runs sorted by finish time (oldest first)
    run_ages = []
    for run_id in os.listdir(RUNS_DIR):
        run_dir = os.path.join(RUNS_DIR, run_id)
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.isdir(run_dir) or not os.path.exists(status_path):
            continue
        try:
            status = json.loads(open(status_path, "r").read())
            if status.get("status") not in ("succeeded", "failed"):
                continue
            finished_at = status.get("finished_at", "")
            run_ages.append((finished_at, run_dir))
        except Exception:
            continue

    run_ages.sort(key=lambda x: x[0])  # oldest first

    for _, run_dir in run_ages:
        if _get_dir_size_mb(RUNS_DIR) <= MAX_STORAGE_MB:
            break
        try:
            shutil.rmtree(run_dir)
        except Exception:
            pass


def _cleanup_stale_processes():
    """P2.2: Kill processes that exceed timeout and clean up finished ones."""
    now = datetime.utcnow()
    finished_ids = []

    for run_id, info in _active_processes.items():
        proc = info["proc"]
        started_at = info["started_at"]

        # Check if process already finished
        if proc.poll() is not None:
            finished_ids.append(run_id)
            continue

        # Kill if exceeded timeout
        elapsed = (now - started_at).total_seconds()
        if elapsed > RUN_TIMEOUT_SECONDS:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

            # Mark as failed in status
            rdir = _run_dir(run_id)
            status_path = os.path.join(rdir, "status.json")
            if os.path.exists(status_path):
                try:
                    _write_json(status_path, {
                        "run_id": run_id,
                        "status": "failed",
                        "error": f"Process killed: exceeded {RUN_TIMEOUT_SECONDS}s timeout",
                        "updated_at": _utc_now(),
                    })
                except Exception:
                    pass
            finished_ids.append(run_id)

    for rid in finished_ids:
        _active_processes.pop(rid, None)


@app.post("/runs", response_model=RunCreateResponse)
def create_run(req: RunCreateRequest):
    """
    Create a training run. Uses Option 2: spawns a subprocess job.
    Returns immediately with queued status. Poll /runs/{run_id}.
    """
    # Validate dataset exists
    _ = _load_dataset_meta(req.dataset_id)

    run_id = str(uuid.uuid4())
    rdir = _run_dir(run_id)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(_run_artifacts_dir(run_id), exist_ok=True)

    # Pass explainability as string — the explainability modules handle "all" internally
    explainability_value = req.explainability.value if req.explainability else None

    # Write request.json for the runner
    request_obj = {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "model_type": req.model_type.value,
        "task": req.task.value,
        "config": req.config,
        "split": req.split,
        "explainability": explainability_value,
        "target_column": req.target_column,
        "mapping_mode": req.mapping_mode,
        "column_mapping": req.column_mapping.model_dump() if req.column_mapping else None,
        "created_at": _utc_now(),
    }
    _write_json(_run_request_path(run_id), request_obj)

    # Initialize status.json
    status_obj = {
        "run_id": run_id,
        "status": "queued",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_json(_run_status_path(run_id), status_obj)

    # P2.2: Clean up stale/timed-out processes before spawning new one
    _cleanup_stale_processes()
    # P2.3: Enforce storage limits before spawning
    _cleanup_old_runs()
    _enforce_disk_quota()

    # Spawn subprocess job (logs to logs.txt)
    log_path = os.path.join(rdir, "logs.txt")
    with open(log_path, "a", encoding="utf-8") as log:
        # Important: run module with repo root as cwd so "backend.*" imports resolve
        proc = subprocess.Popen(
            [PYTHON_EXEC, "-m", "backend.runner.run_job", "--run-dir", rdir],
            stdout=log,
            stderr=log,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    # P2.2: Track process for timeout management
    _active_processes[run_id] = {
        "proc": proc,
        "started_at": datetime.utcnow(),
        "pid": proc.pid,
    }

    # Save pid
    status_obj["pid"] = proc.pid
    status_obj["updated_at"] = _utc_now()
    _write_json(_run_status_path(run_id), status_obj)

    return RunCreateResponse(run_id=run_id, status="queued")


@app.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str):
    """
    Poll run status.
    """
    status = _load_run_status(run_id)
    return RunStatus(**status)


@app.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str, tail: int = 50):
    """
    Fetch last N lines from the run logs.
    """
    rdir = _run_dir(run_id)
    log_path = os.path.join(rdir, "logs.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Run logs not found")

    tail = max(1, min(int(tail), 500))
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    lines = deque(maxlen=tail)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = ansi_re.sub("", line).replace("\r", "").rstrip("\n")
            lines.append(clean)

    return {"run_id": run_id, "lines": list(lines)}


@app.get("/runs/{run_id}/logs.txt")
def download_run_logs(run_id: str):
    """
    Download the raw run log file.
    """
    rdir = _run_dir(run_id)
    log_path = os.path.join(rdir, "logs.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Run logs not found")

    return FileResponse(log_path, filename=f"run_{run_id}_logs.txt", media_type="text/plain")


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str):
    """
    Lists artifacts generated by the run.
    """
    rdir = _run_dir(run_id)
    if not os.path.exists(rdir):
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts_dir = _run_artifacts_dir(run_id)
    if not os.path.exists(artifacts_dir):
        return {"run_id": run_id, "artifacts": []}

    artifacts = []
    for root, _, files in os.walk(artifacts_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, artifacts_dir)
            artifacts.append(rel)

    artifacts.sort()
    return {"run_id": run_id, "artifacts": artifacts}


@app.get("/runs/{run_id}/artifacts/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str):
    """
    Download/view a single artifact file.
    """
    artifacts_dir = _run_artifacts_dir(run_id)
    full = os.path.normpath(os.path.join(artifacts_dir, artifact_path))

    # Prevent path traversal
    if not full.startswith(os.path.abspath(artifacts_dir) + os.sep) and os.path.abspath(full) != os.path.abspath(artifacts_dir):
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(full)


@app.get("/runs/{run_id}/artifacts.zip")
def download_artifacts_zip(run_id: str):
    """
    Download all artifacts as a ZIP archive.
    """
    rdir = _run_dir(run_id)
    if not os.path.exists(rdir):
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts_dir = _run_artifacts_dir(run_id)
    if not os.path.exists(artifacts_dir):
        raise HTTPException(status_code=404, detail="Artifacts not found")

    zip_path = os.path.join(rdir, "artifacts.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(artifacts_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, artifacts_dir)
                zf.write(full, arcname=rel)

    return FileResponse(zip_path, filename=f"run_{run_id}_artifacts.zip")


@app.post("/admin/cleanup")
def admin_cleanup():
    """P2.3: Manual storage cleanup endpoint."""
    removed_runs = _cleanup_old_runs()
    _enforce_disk_quota()
    _cleanup_stale_processes()

    storage_mb = _get_dir_size_mb(RUNS_DIR) if os.path.isdir(RUNS_DIR) else 0
    return {
        "removed_runs": removed_runs,
        "active_processes": len(_active_processes),
        "storage_mb": round(storage_mb, 2),
        "quota_mb": MAX_STORAGE_MB,
        "ttl_hours": RUN_TTL_HOURS,
    }
