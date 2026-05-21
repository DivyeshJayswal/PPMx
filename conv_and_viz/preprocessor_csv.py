import pandas as pd
import numpy as np
import os
from conv_and_viz.xes_to_csv import load_event_log, log_to_dataframe_preserve_all
from utils.column_detector import detect_and_standardize_columns

try:
    import pm4py
    from pm4py.objects.conversion.log import converter as log_converter
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False


def detect_column_type(series):
    """Detect whether a column is categorical, numerical, or datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_bool_dtype(series):
        return 'categorical'
    if pd.api.types.is_numeric_dtype(series):
        return 'numerical'
    return 'categorical'


def preprocess_event_log(input_path, output_csv_path="preprocessed_log.csv", options=None):
    """
    Preprocess an event log file (XES or CSV) for predictive process monitoring.
    
    Args:
        input_path: Path to input file (XES or CSV)
        output_csv_path: Path for output preprocessed CSV
        
    Returns:
        DataFrame: Preprocessed dataframe
    """
    default_options = {
        "sort_and_normalize_timestamps": True,
        "check_millisecond_order": True,
        "impute_categorical": True,
        "impute_numeric_neighbors": True,
        "drop_missing_timestamps": True,
        "fill_remaining_missing": True,
        "remove_duplicates": True,
    }
    if options:
        for key, value in options.items():
            if key in default_options and value is not None:
                default_options[key] = bool(value)

    lower_input_path = input_path.lower()
    if lower_input_path.endswith(".xes") or lower_input_path.endswith(".xes.gz"):
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py is required to process XES files. Install with: pip install pm4py")
        log = load_event_log(input_path)
        df = log_to_dataframe_preserve_all(log)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} events from {input_path}")
    print(f"Original columns: {list(df.columns)}")

    # Only auto-detect timestamp column (no other column detection).
    timestamp_col = None
    case_col = None
    timestamp_patterns = [
        "time:timestamp",
        "timestamp",
        "time",
        "start_time",
        "starttime",
        "complete_time",
        "completetime",
    ]
    for col in df.columns:
        if col.strip().lower() in timestamp_patterns:
            timestamp_col = col
            break

    if timestamp_col and (
        default_options["sort_and_normalize_timestamps"]
        or default_options["check_millisecond_order"]
    ):
        # P1.3: Parse timestamps and coerce errors to NaT
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        nat_count = df[timestamp_col].isna().sum()
        if nat_count > 0 and default_options["drop_missing_timestamps"]:
            print(f"[P1.3] Found {nat_count} NaT values in timestamp column. Will drop these rows.")
            df = df.dropna(subset=[timestamp_col])
            print(f"[P1.3] Dropped {nat_count} rows with invalid timestamps. Remaining rows: {len(df)}")

    if case_col and timestamp_col and default_options["check_millisecond_order"]:
        df['Timestamp_Sec'] = df[timestamp_col].dt.floor('s')
        df['Prev_CaseID'] = df[case_col].shift(1)
        df['Prev_Timestamp_Sec'] = df['Timestamp_Sec'].shift(1)
        
        same_second_mask = (df[case_col] == df['Prev_CaseID']) & \
                           (df['Timestamp_Sec'] == df['Prev_Timestamp_Sec'])
        
        order_critical_mask = same_second_mask & (df[timestamp_col] != df[timestamp_col].shift(1))
        cases_affected_by_milliseconds = df[order_critical_mask][case_col].nunique()
        
        print(f"Total Cases in Log: {df[case_col].nunique()}")
        print(f"Total Unique Cases where Milliseconds Determine Order: {cases_affected_by_milliseconds}")

        df.drop(columns=['Timestamp_Sec', 'Prev_CaseID', 'Prev_Timestamp_Sec'], inplace=True)

    if timestamp_col and default_options["sort_and_normalize_timestamps"]:
        df[timestamp_col] = df[timestamp_col].dt.floor('s')
        if df[timestamp_col].dt.tz is not None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
        # Drop timezone offsets like +00:00 in string representation
        df[timestamp_col] = df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n--- Checking Data Types and Handling Wrong Values ---")
    
    if default_options["impute_categorical"] or default_options["impute_numeric_neighbors"]:
        for col in df.columns:
            if col in [case_col, timestamp_col]:
                continue
            
            detected_type = detect_column_type(df[col])
            
            if detected_type == 'categorical' and default_options["impute_categorical"]:
                df[col] = df[col].ffill().bfill()
            
            elif detected_type == 'numerical' and default_options["impute_numeric_neighbors"]:
                mask = df[col].isna()
                for idx in df[mask].index:
                    prev_idx = idx - 1
                    next_idx = idx + 1
                    
                    prev_val = df.loc[prev_idx, col] if prev_idx in df.index else np.nan
                    next_val = df.loc[next_idx, col] if next_idx in df.index else np.nan
                    
                    if pd.notna(prev_val) and pd.notna(next_val):
                        df.loc[idx, col] = (prev_val + next_val) / 2
                    elif pd.notna(prev_val):
                        df.loc[idx, col] = prev_val
                    elif pd.notna(next_val):
                        df.loc[idx, col] = next_val

    print("\n--- Handling Null and Zero Values ---")
    missing_cols = []
    if default_options["drop_missing_timestamps"] or default_options["fill_remaining_missing"]:
        missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        missing_before = df[missing_cols].isnull().sum()
        print(missing_before[missing_before > 0])
        
        numeric_cols = [col for col in missing_cols if detect_column_type(df[col]) == 'numerical']
        categorical_cols = [col for col in missing_cols if detect_column_type(df[col]) == 'categorical']
        datetime_cols = [col for col in missing_cols if detect_column_type(df[col]) == 'datetime']
        
        if default_options["drop_missing_timestamps"] and timestamp_col and timestamp_col in missing_cols:
            rows_before = len(df)
            df = df.dropna(subset=[timestamp_col])
            rows_after = len(df)
            if rows_before > rows_after:
                print(f"Removed {rows_before - rows_after} rows with missing timestamps")
            missing_cols.remove(timestamp_col)
            if timestamp_col in numeric_cols:
                numeric_cols.remove(timestamp_col)
            if timestamp_col in categorical_cols:
                categorical_cols.remove(timestamp_col)
        
        if default_options["fill_remaining_missing"]:
            for col in numeric_cols:
                df[col] = df[col].fillna(0.0)
            for col in categorical_cols:
                df[col] = df[col].fillna('N/A')
            # Do not fill datetime columns with strings.
        
        print("\n--- Missing Value Count AFTER Imputation ---")
        remaining_missing = [col for col in missing_cols if col in df.columns]
        if remaining_missing:
            missing_after = df[remaining_missing].isnull().sum()
            print(missing_after)
    else:
        print("No missing values found.")

    initial_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    
    print(f"\n--- Looking for duplicates ---")
    print(f"Total rows before checking: {initial_rows}")
    print(f"Total exact duplicate rows found: {duplicate_rows}")
    
    if default_options["remove_duplicates"] and duplicate_rows > 0:
        df.drop_duplicates(inplace=True) 
        
        final_rows = len(df)
        print(f"Removed {duplicate_rows} duplicate rows.")
        print(f"Total rows after removal: {final_rows}")
    elif duplicate_rows == 0:
        print("No exact duplicate rows found.")
    else:
        print("Duplicate rows found, but removal is disabled.")
    
    # P1.4: Detect and remove zero-variance columns
    print("\n--- Checking for Zero-Variance Columns ---")
    zero_variance_cols = []
    for col in df.columns:
        if col in ['CaseID', 'Activity', 'Timestamp', 'Resource']:
            # Skip core event log columns
            continue
        unique_count = df[col].nunique()
        if unique_count <= 1:
            zero_variance_cols.append(col)

    if zero_variance_cols:
        print(f"[P1.4] Found {len(zero_variance_cols)} zero-variance columns: {zero_variance_cols}")
        df = df.drop(columns=zero_variance_cols)
        print(f"[P1.4] Removed zero-variance columns. Remaining columns: {len(df.columns)}")
    else:
        print("[P1.4] No zero-variance columns found.")

    # P1.1: Validate column structure and minimum event/case counts
    try:
        df, mapping, detected = detect_and_standardize_columns(df, verbose=True, min_cases=2, min_events=10)
    except ValueError as e:
        print(f"[X] Column validation failed: {e}")
        raise

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv_path, index=False)
    print(f"\n[OK] Preprocessed file saved at: {output_csv_path}")

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def main():
    input_file = "BPI_Models/BPI_logs_csv/BPI_2020_Log_PermitLog.csv"
    output_file = "BPI_Models/BPI_logs_preprocessed_csv/BPI_2020_Log_PermitLog_preprocessed.csv"
    
    if not os.path.exists(input_file):
        print(f"[X] Input file not found: {input_file}")
        print("Please update the path to your input file.")
        return
    
    df = preprocess_event_log(
        input_path=input_file,
        output_csv_path=output_file
    )
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
