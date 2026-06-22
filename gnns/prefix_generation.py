import os
import sys
from pathlib import Path
import pandas as pd


def detect_columns(df: pd.DataFrame):
    col_map = {}
    
    if 'time:timestamp' in df.columns:
        col_map['time:timestamp'] = 'Timestamp'
    elif 'Timestamp' in df.columns:
        col_map['Timestamp'] = 'Timestamp'
    
    if 'case:id' in df.columns:
        col_map['case:id'] = 'CaseID'
    elif 'CaseID' in df.columns:
        col_map['CaseID'] = 'CaseID'
    
    if 'concept:name' in df.columns:
        col_map['concept:name'] = 'Activity'
    elif 'Activity' in df.columns:
        col_map['Activity'] = 'Activity'
    
    if 'org:resource' in df.columns:
        col_map['org:resource'] = 'Resource'
    elif 'org:group' in df.columns:
        col_map['org:group'] = 'Resource'
    elif 'Resource' in df.columns:
        col_map['Resource'] = 'Resource'
    
    return col_map


def load_and_sort_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    col_map = detect_columns(df)
    rename_map = {k: v for k, v in col_map.items() if k != v}
    
    if rename_map:
        print(f"Renaming columns: {list(rename_map.keys())}")
        df = df.rename(columns=rename_map)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)
    return df


def detect_trace_attributes(df: pd.DataFrame):
    trace_cols = []
    grouped = df.groupby('CaseID')

    for col in df.columns:
        if col == 'CaseID':
            continue
        if grouped[col].nunique().max() == 1:
            trace_cols.append(col)

    return trace_cols



def generate_prefix_dataset(df: pd.DataFrame, trace_cols):
    rows = []

    for case_id, group in df.groupby('CaseID'):
        group = group.sort_values('Timestamp').reset_index(drop=True)
        trace_len = len(group)

        trace_attrs = {c: group[c].iloc[0] for c in trace_cols}

        for k in range(1, trace_len):
            label_next_activity = group.iloc[k]["Activity"]

            prefix = group.iloc[:k]

            for pos, (_, ev) in enumerate(prefix.iterrows(), start=1):
                row = {
                    "CaseID": case_id,
                    "prefix_id": k,
                    "prefix_pos": pos,
                    "prefix_length": k,
                    "Activity": ev["Activity"],
                    "Resource": ev["Resource"],
                    "Timestamp": ev["Timestamp"],
                    "next_activity": label_next_activity,
                }

                row.update(trace_attrs)
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("This is a standalone utility script.")
    print("Prefix generation is now handled in-memory by gnn_predictor.py")
    print("No files are created automatically.")


if __name__ == "__main__":
    main()
