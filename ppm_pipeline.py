# ppm_pipeline.py
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

np.random.seed(42)

EXPLAINABILITY_AVAILABLE = True
EXPLAINABILITY_IMPORT_ERROR = None
try:
    from explainability.transformers import run_transformer_explainability
    from explainability.gnns import run_gnn_explainability
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    EXPLAINABILITY_IMPORT_ERROR = str(e)

# We already verified TF/Torch are installed, but keep these guards for robustness
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    torch.manual_seed(42)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

if TENSORFLOW_AVAILABLE:
    from transformers.prediction.next_activity import NextActivityPredictor
    from transformers.prediction.event_time import EventTimePredictor
    from transformers.prediction.remaining_time import RemainingTimePredictor

if PYTORCH_AVAILABLE:
    from gnns.prediction.gnn_predictor import GNNPredictor


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _positive_case_value(target_series):
    if target_series is None:
        return "N/A"
    series = pd.Series(target_series).dropna()
    unique_values = sorted(series.astype(str).unique().tolist())
    if len(unique_values) != 2:
        return "N/A"
    positive_hints = {"1", "1.0", "true", "yes", "positive", "completed"}
    chosen = None
    for value in unique_values:
        if value.strip().lower() in positive_hints:
            chosen = value
            break
    if chosen is None:
        chosen = unique_values[-1]
    proportion = float((series.astype(str) == chosen).mean())
    return f"{proportion * 100:.1f}% ({chosen})"


def _describe_run(model_type, task):
    task_label = str(task).replace("_", " ")
    if model_type == "transformer":
        return f"Transformer-based predictive process monitoring for {task_label}."
    return f"Heterogeneous graph predictive process monitoring for {task_label}."


def _build_dataset_summary_rows(model_type, task, dataset_path, df, case_col, data, config, target_series=None, dataset_display_name=None):
    case_lengths = df.groupby(case_col).size() if case_col in df.columns else pd.Series(dtype=int)
    num_cases = int(case_lengths.shape[0]) if not case_lengths.empty else 0
    max_prefix_length = int(max(case_lengths.max() - 1, 0)) if not case_lengths.empty else 0
    observed_prefix_min = 1 if max_prefix_length > 0 else 0
    if model_type == "transformer":
        used_prefix_max = min(max_prefix_length, int(config.get("max_len", max_prefix_length or 0)))
    else:
        used_prefix_max = max_prefix_length

    generated_samples = None
    feature_single = "N/A"
    feature_bucket_agg = "N/A"
    feature_bucket_index = "N/A"

    if model_type == "transformer":
        if "X_train" in data:
            generated_samples = int(len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]))
            feature_bucket_index = (
                f"Train tensor: {tuple(data['X_train'].shape)} | "
                f"Val: {tuple(data['X_val'].shape)} | Test: {tuple(data['X_test'].shape)}"
            )
        elif "X_seq_train" in data:
            generated_samples = int(len(data["X_seq_train"]) + len(data["X_seq_val"]) + len(data["X_seq_test"]))
            feature_single = (
                f"Temporal feature vector: {tuple(data['X_temp_train'].shape)}"
                if "X_temp_train" in data
                else "N/A"
            )
            feature_bucket_agg = (
                f"Time sequence tensor: {tuple(data['X_time_train'].shape)}"
                if "X_time_train" in data
                else "N/A"
            )
            feature_bucket_index = f"Activity sequence tensor: {tuple(data['X_seq_train'].shape)}"
    else:
        train_graphs = data.get("train", [])
        val_graphs = data.get("val", [])
        test_graphs = data.get("test", [])
        all_graphs = list(train_graphs) + list(val_graphs) + list(test_graphs)
        generated_samples = int(len(all_graphs))
        if all_graphs:
            sample_graph = all_graphs[0]
            node_counts = [int(g["activity"].num_nodes) for g in all_graphs if "activity" in g.node_types]
            min_nodes = min(node_counts) if node_counts else 0
            max_nodes = max(node_counts) if node_counts else 0
            feature_single = f"Trace vector dim: {tuple(sample_graph['trace'].x.shape)}"
            feature_bucket_agg = (
                f"Node feature dims: activity={sample_graph['activity'].x.shape[1]}, "
                f"resource={sample_graph['resource'].x.shape[1]}, time={sample_graph['time'].x.shape[1]}"
            )
            feature_bucket_index = f"Graph prefixes: {len(all_graphs)} | Activity nodes min/max: {min_nodes} / {max_nodes}"

    prefix_range = "N/A"
    if used_prefix_max > 0:
        prefix_range = f"{observed_prefix_min} - {used_prefix_max}"

    rows = [
        {"label": "Event Log", "value": dataset_display_name or os.path.basename(dataset_path)},
        {"label": "Description", "value": _describe_run(model_type, task)},
        {"label": "No. of Cases (before encoding)", "value": num_cases},
        {"label": "Proportion of Positive Cases", "value": _positive_case_value(target_series)},
        {"label": "Maximum Prefix Length", "value": max_prefix_length if max_prefix_length > 0 else "N/A"},
        {"label": "Prefix Lengths Used", "value": prefix_range},
        {"label": "Single Bucket & Aggregate Encoding", "value": feature_single},
        {"label": "Prefix-length buckets & aggregate encoding", "value": feature_bucket_agg},
        {"label": "Prefix-length buckets & Index-Based Encoding", "value": feature_bucket_index},
    ]
    if generated_samples is not None:
        rows.append({"label": "Generated Training Samples", "value": generated_samples})
    return rows


def _build_training_summary_rows(model_type, predictor, config, metrics):
    rows = [
        {"label": "Model Type", "value": model_type},
        {"label": "Epochs Requested", "value": config.get("epochs")},
        {"label": "Batch Size", "value": config.get("batch_size")},
        {"label": "Patience", "value": config.get("patience")},
    ]
    if "lr" in config:
        rows.append({"label": "Learning Rate", "value": config.get("lr")})
    rows.append({"label": "Dropout Rate", "value": config.get("dropout_rate")})

    if model_type == "transformer":
        rows.extend(
            [
                {"label": "Max Sequence Length", "value": config.get("max_len")},
                {"label": "Model Dimension", "value": config.get("d_model")},
                {"label": "Attention Heads", "value": config.get("num_heads")},
                {"label": "Transformer Blocks", "value": config.get("num_blocks")},
            ]
        )
        history = getattr(getattr(predictor, "history", None), "history", None)
        if isinstance(history, dict) and history:
            first_metric = next((key for key in history.keys() if key.startswith("val_")), None)
            loss_key = "loss" if "loss" in history else None
            rows.append({"label": "Epochs Completed", "value": len(next(iter(history.values())))})
            if loss_key:
                rows.append({"label": "Final Training Loss", "value": float(history[loss_key][-1])})
            if "val_loss" in history:
                rows.append({"label": "Best Validation Loss", "value": float(min(history["val_loss"]))})
            if "accuracy" in history:
                rows.append({"label": "Final Training Accuracy", "value": float(history["accuracy"][-1])})
            if "val_accuracy" in history:
                rows.append({"label": "Best Validation Accuracy", "value": float(max(history["val_accuracy"]))})
            if "mae" in history:
                rows.append({"label": "Final Training MAE", "value": float(history["mae"][-1])})
            if "val_mae" in history:
                rows.append({"label": "Best Validation MAE", "value": float(min(history["val_mae"]))})
            if first_metric and first_metric not in {"val_loss", "val_accuracy", "val_mae"}:
                rows.append({"label": f"Tracked Validation Metric ({first_metric})", "value": float(history[first_metric][-1])})
    else:
        rows.extend(
            [
                {"label": "Hidden Channels", "value": config.get("hidden")},
                {"label": "Attention Heads", "value": config.get("heads")},
                {"label": "GAT Layers", "value": config.get("num_layers")},
            ]
        )
        history = getattr(predictor, "history", None)
        if isinstance(history, dict) and history:
            rows.append({"label": "Epochs Completed", "value": len(history.get("train_loss", []))})
            if history.get("train_loss"):
                rows.append({"label": "Final Training Loss", "value": float(history["train_loss"][-1])})
            if history.get("val_loss"):
                rows.append({"label": "Best Validation Loss", "value": float(min(history["val_loss"]))})
            if history.get("train_acc"):
                rows.append({"label": "Final Training Accuracy", "value": float(history["train_acc"][-1])})
            if history.get("val_acc"):
                rows.append({"label": "Best Validation Accuracy", "value": float(max(history["val_acc"]))})
            if history.get("val_mae_time"):
                rows.append({"label": "Best Validation Event Time MAE", "value": float(min(history["val_mae_time"]))})
            if history.get("val_mae_rem"):
                rows.append({"label": "Best Validation Remaining Time MAE", "value": float(min(history["val_mae_rem"]))})

    if isinstance(metrics, dict):
        for key, label in [
            ("test_accuracy", "Test Accuracy"),
            ("accuracy", "Test Accuracy"),
            ("test_loss", "Test Loss"),
            ("loss", "Test Loss"),
            ("test_mae", "Test MAE"),
            ("mae_time", "Test Event Time MAE"),
            ("mae_rem", "Test Remaining Time MAE"),
        ]:
            if key in metrics:
                rows.append({"label": label, "value": metrics[key]})

    return rows


def _write_run_summaries(output_dir, dataset_rows, training_rows):
    _write_json(os.path.join(output_dir, "dataset_summary.json"), {"rows": dataset_rows})
    _write_json(os.path.join(output_dir, "training_summary.json"), {"rows": training_rows})


def _mae_against_constant(y_true, constant_value):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if y_true.size == 0:
        return None
    preds = np.full_like(y_true, float(constant_value), dtype=float)
    return float(np.mean(np.abs(y_true - preds)))


def _add_regression_baselines(metrics, train_targets, test_targets, metric_prefix):
    train_targets = np.asarray(train_targets, dtype=float).reshape(-1)
    test_targets = np.asarray(test_targets, dtype=float).reshape(-1)
    if train_targets.size == 0 or test_targets.size == 0:
        return metrics

    mean_value = float(np.mean(train_targets))
    median_value = float(np.median(train_targets))
    mean_mae = _mae_against_constant(test_targets, mean_value)
    median_mae = _mae_against_constant(test_targets, median_value)

    metrics[f"{metric_prefix}_baseline_mean_value"] = mean_value
    metrics[f"{metric_prefix}_baseline_median_value"] = median_value
    metrics[f"{metric_prefix}_baseline_mean_mae"] = mean_mae
    metrics[f"{metric_prefix}_baseline_median_mae"] = median_mae
    return metrics


def _extract_graph_targets(graphs, attr_name):
    values = []
    for graph in graphs:
        target = getattr(graph, attr_name, None)
        if target is None:
            continue
        values.append(float(target.view(-1)[0].item()))
    return np.asarray(values, dtype=float)


def detect_and_standardize_columns(df, verbose=False):
    column_mapping = {}

    case_patterns = ['case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    activity_patterns = ['concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
    timestamp_patterns = ['time:timestamp', 'Timestamp', 'timestamp', 'time', 'Time', 'start_time', 'StartTime', 'complete_time', 'CompleteTime']
    resource_patterns = ['org:resource', 'Resource', 'resource', 'user', 'User', 'org:role', 'role', 'Role', 'actor', 'Actor']

    for col in df.columns:
        if col in case_patterns and col != 'CaseID':
            column_mapping[col] = 'CaseID'
            break

    # Activity - only map if 'Activity' doesn't already exist
    if 'Activity' not in df.columns:
        for col in df.columns:
            if col in activity_patterns:
                column_mapping[col] = 'Activity'
                if verbose:
                    print(f"[COLUMN DETECT] Mapping '{col}' -> 'Activity'")
                break
    else:
        if verbose:
            print(f"[COLUMN DETECT] Using existing 'Activity' column")

    # Timestamp - only map if 'Timestamp' doesn't already exist
    if 'Timestamp' not in df.columns:
        for col in df.columns:
            if col in timestamp_patterns:
                column_mapping[col] = 'Timestamp'
                break

    # Resource - only map if 'Resource' doesn't already exist  
    if 'Resource' not in df.columns:
        for col in df.columns:
            if col in resource_patterns:
                column_mapping[col] = 'Resource'
                break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ['CaseID', 'Activity', 'Timestamp']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")

    return df, column_mapping, column_mapping.keys()


def _safe_rename_columns(df, rename_map):
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    for src, tgt in rename_map.items():
        if tgt in df.columns and tgt != src and tgt not in rename_map.keys():
            df = df.drop(columns=[tgt])
    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def default_transformer_config():
    return {
        'max_len': 16,
        'd_model': 64,
        'num_heads': 4,
        'num_blocks': 2,
        'dropout_rate': 0.1,
        'epochs': 5,
        'batch_size': 128,
        'patience': 10
    }


def default_gnn_config():
    return {
        'hidden': 64,
        'heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'lr': 4e-4,
        'epochs': 5,
        'batch_size': 64,
        'patience': 10
    }


def run_next_activity_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    explainability_config=None,
    target_column=None,
    skip_auto_mapping=False,
    dataset_display_name=None,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    
    # DEBUG 1: Raw dataset
    if 'Activity' in df.columns:
        print("\n[DEBUG 1] RAW CSV:", df['Activity'].nunique(), "unique activities")
        print("Sample:", list(df['Activity'].unique()[:5]))
    else:
        print("\n[DEBUG 1] RAW CSV: Activity column not found")
        print("Columns:", list(df.columns))
    
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    target_series = None
    if target_column:
        if target_column not in df.columns:
            raise RuntimeError(f"Target column not found: {target_column}")
        if pd.api.types.is_numeric_dtype(df[target_column]):
            raise RuntimeError("Invalid target column selected: must be categorical.")
        target_series = df[target_column].astype(str)
    
    # DEBUG 2: After standardization
    if 'Activity' in df.columns:
        print("[DEBUG 2] AFTER STANDARDIZE:", df['Activity'].nunique(), "activities")
        print("Sample:", list(df['Activity'].unique()[:5]))
    else:
        print("[DEBUG 2] ERROR: Activity column missing after standardization")
        print("Columns:", list(df.columns))

    df = _safe_rename_columns(df, {
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    if target_series is not None:
        df['concept:name'] = target_series

    predictor = NextActivityPredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(
        df,
        test_size=test_size,
        val_split=val_split,
        max_cases=config.get("max_cases"),
        max_prefixes_per_case=config.get("max_prefixes_per_case"),
        max_graphs=config.get("max_graphs"),
    )
    
    # DEBUG 3: After prepare_data
    print("[DEBUG 3] LABEL ENCODER:", len(predictor.label_encoder.classes_), "classes")
    print("Classes:", list(predictor.label_encoder.classes_))
    
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    y_pred, y_pred_probs = predictor.predict(data)
    predictor.save_results(data, y_pred, y_pred_probs, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    _write_run_summaries(
        output_dir,
        _build_dataset_summary_rows(
            "transformer",
            "next_activity" if target_column is None else "custom_activity",
            dataset_path,
            df,
            "case:id",
            data,
            config,
            target_series=target_series,
            dataset_display_name=dataset_display_name,
        ),
        _build_training_summary_rows("transformer", predictor, config, metrics),
    )

    if explainability_method and not EXPLAINABILITY_AVAILABLE:
        raise RuntimeError(
            "Explainability requested, but explainability modules are unavailable: "
            f"{EXPLAINABILITY_IMPORT_ERROR or 'unknown import error'}"
        )

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_config = explainability_config or {}
        explainability_dir = os.path.join(output_dir, 'explainability')
        explainability_samples = explainability_config.get(
            "transformer_explanation_samples",
            config.get("explainability_samples", 50),
        )
        feature_config = {}
        if hasattr(predictor, "vocab_size") and predictor.vocab_size is not None:
            feature_config["vocab_size"] = predictor.vocab_size

        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='activity',
            num_samples=explainability_samples,
            methods=explainability_method,
            label_encoder=predictor.label_encoder,
            scaler=getattr(predictor, 'scaler', None),
            feature_config=feature_config,
            benchmark_config=explainability_config,
            local_num_samples=explainability_config.get("local_explanation_samples") or None,
            global_sample_percent=explainability_config.get("global_explanation_sample_percent", 100),
            min_prefix_length=explainability_config.get("min_prefix_length") or None,
            max_prefix_length=explainability_config.get("max_prefix_length") or None,
        )

    return metrics


def run_event_time_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    explainability_config=None,
    skip_auto_mapping=False,
    dataset_display_name=None,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    predictor = EventTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    metrics = _add_regression_baselines(
        metrics,
        data.get('y_train', []),
        data.get('y_test', []),
        'event_time',
    )
    y_pred = predictor.predict(data)
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    _write_run_summaries(
        output_dir,
        _build_dataset_summary_rows(
            "transformer",
            "event_time",
            dataset_path,
            df,
            "case:concept:name",
            data,
            config,
            dataset_display_name=dataset_display_name,
        ),
        _build_training_summary_rows("transformer", predictor, config, metrics),
    )

    if explainability_method and not EXPLAINABILITY_AVAILABLE:
        raise RuntimeError(
            "Explainability requested, but explainability modules are unavailable: "
            f"{EXPLAINABILITY_IMPORT_ERROR or 'unknown import error'}"
        )

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_config = explainability_config or {}
        explainability_dir = os.path.join(output_dir, 'explainability')
        explainability_samples = explainability_config.get(
            "transformer_explanation_samples",
            config.get("explainability_samples", 50),
        )
        feature_config = {}
        if hasattr(predictor, "vocab_size") and predictor.vocab_size is not None:
            feature_config["vocab_size"] = predictor.vocab_size

        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=explainability_samples,
            methods=explainability_method,
            label_encoder=predictor.label_encoder,
            scaler=predictor.scaler,
            feature_config=feature_config,
            timestamps=data.get('X_time_test'),
            benchmark_config=explainability_config,
            local_num_samples=explainability_config.get("local_explanation_samples") or None,
            global_sample_percent=explainability_config.get("global_explanation_sample_percent", 100),
            min_prefix_length=explainability_config.get("min_prefix_length") or None,
            max_prefix_length=explainability_config.get("max_prefix_length") or None,
        )

    return metrics


def run_remaining_time_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    explainability_config=None,
    skip_auto_mapping=False,
    dataset_display_name=None,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    predictor = RemainingTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    metrics = _add_regression_baselines(
        metrics,
        data.get('y_train', []),
        data.get('y_test', []),
        'remaining_time',
    )
    y_pred = predictor.predict(data)
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    _write_run_summaries(
        output_dir,
        _build_dataset_summary_rows(
            "transformer",
            "remaining_time",
            dataset_path,
            df,
            "case:concept:name",
            data,
            config,
            dataset_display_name=dataset_display_name,
        ),
        _build_training_summary_rows("transformer", predictor, config, metrics),
    )

    if explainability_method and not EXPLAINABILITY_AVAILABLE:
        raise RuntimeError(
            "Explainability requested, but explainability modules are unavailable: "
            f"{EXPLAINABILITY_IMPORT_ERROR or 'unknown import error'}"
        )

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_config = explainability_config or {}
        explainability_dir = os.path.join(output_dir, 'explainability')
        explainability_samples = explainability_config.get(
            "transformer_explanation_samples",
            config.get("explainability_samples", 50),
        )
        feature_config = {}
        if hasattr(predictor, "vocab_size") and predictor.vocab_size is not None:
            feature_config["vocab_size"] = predictor.vocab_size

        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=explainability_samples,
            methods=explainability_method,
            label_encoder=predictor.label_encoder,
            scaler=predictor.scaler,
            feature_config=feature_config,
            timestamps=data.get('X_time_test'),
            benchmark_config=explainability_config,
            local_num_samples=explainability_config.get("local_explanation_samples") or None,
            global_sample_percent=explainability_config.get("global_explanation_sample_percent", 100),
            min_prefix_length=explainability_config.get("min_prefix_length") or None,
            max_prefix_length=explainability_config.get("max_prefix_length") or None,
        )

    return metrics


def run_gnn_unified_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    explainability_config=None,
    task='unified',
    target_column=None,
    skip_auto_mapping=False,
    dataset_display_name=None,
):
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. GNN runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)
    if target_column:
        if target_column not in df.columns:
            raise RuntimeError(f"Target column not found: {target_column}")
        if pd.api.types.is_numeric_dtype(df[target_column]):
            raise RuntimeError("Invalid target column selected: must be categorical.")
        df["Activity"] = df[target_column].astype(str)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)

    if task == 'unified':
        loss_weights = (1.0, 0.1, 0.1)
    elif task == 'next_activity':
        loss_weights = (1.0, 0.0, 0.0)
    elif task == 'event_time':
        loss_weights = (0.0, 1.0, 0.0)
    elif task == 'remaining_time':
        loss_weights = (0.0, 0.0, 1.0)
    else:
        loss_weights = (1.0, 0.1, 0.1)

    predictor = GNNPredictor(
        hidden_channels=config.get('hidden', 64),
        dropout=config.get('dropout_rate', 0.1),
        lr=config.get('lr', 4e-4),
        loss_weights=loss_weights,
        num_layers=config.get('num_layers', 2),
        heads=config.get('heads', 4),
    )
    predictor.task_name = task

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)

    predictor.build_model(
        data['sample_graph'],
        batch_size=config.get('batch_size', 64),
        num_activity_classes=data.get('num_activity_classes')
    )

    predictor.train(
        data,
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 64),
        patience=config.get('patience', 10)
    )

    metrics = predictor.evaluate_test(data, batch_size=config.get('batch_size', 64))
    if task in {'unified', 'event_time'}:
        metrics = _add_regression_baselines(
            metrics,
            _extract_graph_targets(data.get('train', []), 'y_timestamp'),
            _extract_graph_targets(data.get('test', []), 'y_timestamp'),
            'event_time',
        )
    if task in {'unified', 'remaining_time'}:
        metrics = _add_regression_baselines(
            metrics,
            _extract_graph_targets(data.get('train', []), 'y_remaining_time'),
            _extract_graph_targets(data.get('test', []), 'y_remaining_time'),
            'remaining_time',
        )
    predictor.save_model(output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_results(metrics, output_dir)
    _write_run_summaries(
        output_dir,
        _build_dataset_summary_rows(
            "gnn",
            task,
            dataset_path,
            df,
            "CaseID",
            data,
            config,
            dataset_display_name=dataset_display_name,
        ),
        _build_training_summary_rows("gnn", predictor, config, metrics),
    )

    if explainability_method and not EXPLAINABILITY_AVAILABLE:
        raise RuntimeError(
            "Explainability requested, but explainability modules are unavailable: "
            f"{EXPLAINABILITY_IMPORT_ERROR or 'unknown import error'}"
        )

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_config = explainability_config or {}
        explainability_dir = os.path.join(output_dir, 'explainability')
        if task == 'unified':
            tasks_to_explain = ['activity', 'event_time', 'remaining_time']
        elif task == 'next_activity':
            tasks_to_explain = ['activity']
        elif task == 'event_time':
            tasks_to_explain = ['event_time']
        elif task == 'remaining_time':
            tasks_to_explain = ['remaining_time']
        else:
            tasks_to_explain = ['activity']

        run_gnn_explainability(
            predictor.model,
            data,
            explainability_dir,
            predictor.device,
            vocabularies=data.get('vocabs'),
            num_samples=explainability_config.get('explainability_samples', 10),
            local_num_samples=explainability_config.get('local_explanation_samples', 5),
            methods=explainability_method,
            tasks=tasks_to_explain,
            global_sample_percent=explainability_config.get('global_explanation_sample_percent', 1),
            benchmark_sample_count=explainability_config.get('benchmark_samples'),
            min_prefix_length=explainability_config.get('min_prefix_length'),
            max_prefix_length=explainability_config.get('max_prefix_length'),
        )

    return metrics
