import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm


CASE_COL = "CaseID"
PREFIX_ID_COL = "prefix_id"
PREFIX_POS_COL = "prefix_pos"

IGNORE_COLS = {
    "CaseID", "prefix_id", "prefix_pos", "prefix_length",
    "Activity", "Resource", "Timestamp", "next_activity"
}


def load_prefix_table(path: str) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    print(f"Loaded {len(df)} rows")

    df["__ts_log"] = np.log1p(df["Timestamp"].astype("int64") // 1_000_000_000).astype("float32")

    return df


def detect_trace_attributes(df: pd.DataFrame):
    trace_cols = []
    
    for col in df.columns:
        if col in IGNORE_COLS:
            continue
        
        grouped = df.groupby([CASE_COL, PREFIX_ID_COL])
        col_variance = grouped[col].nunique()
        
        if col_variance.max() == 1:
            trace_cols.append(col)
    
    return trace_cols


def build_global_vocabs(df: pd.DataFrame, trace_attributes: list):
    vocabs = {}

    values = sorted(
        set(df["Activity"].unique().tolist()) |
        set(df["next_activity"].unique().tolist())
    )
    vocabs["Activity"] = {v: i for i, v in enumerate(values)}

    res_vals = sorted(df["Resource"].unique().tolist())
    vocabs["Resource"] = {v: i for i, v in enumerate(res_vals)}

    for col in trace_attributes:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = sorted(df[col].fillna("NaN").unique().tolist())
        vocabs[col] = {v: i for i, v in enumerate(vals)}

    print("\nVocabularies built:")
    for k, v in vocabs.items():
        print(f"  {k}: {len(v)} classes")
    
    return vocabs


def build_dfr_edges(k: int):
    if k < 2:
        return torch.empty((2, 0), dtype=torch.long)
    idx = torch.arange(k)
    return torch.stack([idx[:-1], idx[1:]])


def _signed_log1p(value):
    v = float(value)
    if not np.isfinite(v):
        raise ValueError("Non-finite trace value")
    return np.sign(v) * np.log1p(abs(v))


def build_graph(prefix: pd.DataFrame, vocabs, trace_attributes):
    data = HeteroData()
    k = len(prefix)
    if "CaseID" in prefix.columns:
        data.case_id = str(prefix.iloc[0]["CaseID"])
    if "prefix_id" in prefix.columns:
        data.prefix_id = int(prefix.iloc[0]["prefix_id"])

    act_map = vocabs["Activity"]
    res_map = vocabs["Resource"]

    act_arr = prefix["Activity"].to_numpy()
    res_arr = prefix["Resource"].to_numpy()

    act_ids = np.vectorize(act_map.get)(act_arr)
    data["activity"].x = F.one_hot(
        torch.tensor(act_ids, dtype=torch.long),
        num_classes=len(act_map)
    ).float()

    res_ids = np.vectorize(res_map.get)(res_arr)
    data["resource"].x = F.one_hot(
        torch.tensor(res_ids, dtype=torch.long),
        num_classes=len(res_map)
    ).float()

    data["time"].x = torch.tensor(
        prefix["__ts_log"].to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1)
    prefix_times = pd.to_datetime(prefix["Timestamp"])
    prefix_start = prefix_times.iloc[0]
    relative_seconds = (
        (prefix_times - prefix_start).dt.total_seconds().to_numpy(dtype=np.float32)
    )
    data["time"].display_seconds = torch.tensor(relative_seconds, dtype=torch.float32)

    trace_features = []
    first = prefix.iloc[0]

    for col in trace_attributes:
        if col not in prefix.columns:
            continue
        val = first[col]

        if col in vocabs:
            idx = vocabs[col].get(val, 0)
            trace_features.append(
                F.one_hot(torch.tensor(idx), num_classes=len(vocabs[col])).float()
            )
        else:
            try:
                trace_features.append(torch.tensor([_signed_log1p(val)], dtype=torch.float32))
            except Exception:
                trace_features.append(torch.zeros(1))

    if not trace_features:
        trace_features = [torch.zeros(1)]

    data["trace"].x = torch.cat(trace_features).unsqueeze(0)

    idx = torch.arange(k)
    dfr = build_dfr_edges(k)

    data["activity", "next", "activity"].edge_index = dfr
    data["resource", "next", "resource"].edge_index = dfr.clone()
    data["time", "next", "time"].edge_index = dfr.clone()

    same_ev = torch.stack([idx, idx])
    data["activity", "same_event", "resource"].edge_index = same_ev
    data["resource", "same_event", "activity"].edge_index = same_ev.clone()

    data["activity", "same_time", "time"].edge_index = same_ev.clone()
    data["time", "same_time", "activity"].edge_index = same_ev.clone()

    trace_src = torch.zeros(k, dtype=torch.long)
    data["activity", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
    data["resource", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
    data["time", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])

    next_act = act_map[first["next_activity"]]
    data.y_activity = torch.tensor([next_act], dtype=torch.long)

    if "next_timestamp" in prefix.columns:
        t_next = prefix.iloc[0]["next_timestamp"].timestamp()
    else:
        if k > 1:
            t_next = prefix.iloc[1]["Timestamp"].timestamp()
        else:
            t_next = first["Timestamp"].timestamp()
    data.y_timestamp = torch.tensor([np.log1p(t_next)], dtype=torch.float32)

    if "remaining_time_target" in prefix.columns:
        remaining = float(prefix.iloc[0]["remaining_time_target"])
    else:
        t_end = prefix.iloc[-1]["Timestamp"].timestamp()
        t_now = first["Timestamp"].timestamp()
        remaining = max(0, t_end - t_now)
    data.y_remaining_time = torch.tensor([np.log1p(remaining)], dtype=torch.float32)

    return data


def main():
    print("This is a standalone utility script.")
    print("Graph building is now handled in-memory by gnn_predictor.py")
    print("No files are created automatically.")


if __name__ == "__main__":
    main()
