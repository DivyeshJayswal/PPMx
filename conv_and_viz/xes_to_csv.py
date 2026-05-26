import os
import pandas as pd

from conv_and_viz.xes_utils import normalized_xes_path


def _prepare_pm4py_import() -> None:
    """
    PM4Py imports psutil during module initialization and assumes os.getppid()
    maps to a valid process. In Cloud Run that can be PID 0, which makes
    psutil raise NoSuchProcess before our app can handle the request.
    """
    try:
        import psutil
    except ImportError:
        return

    if getattr(psutil, "_ppm_pid_zero_patch", False):
        return

    original_process = psutil.Process

    def process_with_pid_zero_guard(pid=None):
        if pid == 0:
            pid = os.getpid()
        return original_process(pid)

    psutil.Process = process_with_pid_zero_guard
    psutil._ppm_pid_zero_patch = True


def load_event_log(xes_path: str):
    """Load an XES event log file."""
    _prepare_pm4py_import()
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.log.importer.xes.variants import iterparse as xes_iterparse

    with normalized_xes_path(xes_path) as import_path:
        parameters = {
            xes_iterparse.Parameters.SHOW_PROGRESS_BAR: False,
        }
        return xes_importer.apply(import_path, parameters=parameters)


def log_to_dataframe_preserve_all(event_log):
    """Convert event log to DataFrame using PM4Py's built-in converter to preserve all attributes."""
    _prepare_pm4py_import()
    from pm4py.objects.conversion.log import converter as log_converter

    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df


def log_to_dataframe_manual(event_log):
    """Manually convert event log to DataFrame, preserving trace and event attributes."""
    data = []
    for trace in event_log:
        trace_attrs = dict(trace.attributes)

        for event in trace:
            row = trace_attrs.copy()
            row.update(dict(event))
            data.append(row)

    return pd.DataFrame(data)


def convert_xes_to_csv(xes_path: str, output_folder: str = None):
    """
    Convert XES file to CSV format.

    Args:
        xes_path: Path to the XES file
        output_folder: Output folder for CSV file. If None, uses same folder as XES.

    Returns:
        tuple: (csv_path, dataframe, event_log)
    """
    if output_folder is None:
        output_folder = os.path.dirname(xes_path)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading XES file: {xes_path}")
    event_log = load_event_log(xes_path)

    df = log_to_dataframe_preserve_all(event_log)

    if df.empty or len(df.columns) < 3:
        print("Using manual conversion method...")
        df = log_to_dataframe_manual(event_log)

    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[OK] XES converted to CSV: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Events: {len(df):,}")

    return csv_path, df, event_log


def main():
    xes_path = r"BPI_Models/BPI_logs_xes/BPI_2020_Log_RequestForPayment.xes"
    output_folder = "BPI_Models/BPI_logs_csv"

    if not os.path.exists(xes_path):
        print(f"[X] XES file not found: {xes_path}")
        print("Please update the path to your XES file.")
        return

    csv_path, df, event_log = convert_xes_to_csv(xes_path, output_folder)

    print(f"\nConversion complete!")
    print(f"CSV saved at: {csv_path}")
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
