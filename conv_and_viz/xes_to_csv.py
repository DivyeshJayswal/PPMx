import io
import logging
import os
import sys
from contextlib import contextmanager

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.importer.xes.variants import iterparse as xes_iterparse
from pm4py.objects.conversion.log import converter as log_converter

from conv_and_viz.xes_utils import normalized_xes_path

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_stdout():
    """Redirect stdout/stderr to devnull during PM4Py operations.

    PM4Py prints heavily during XES parsing.  When the backend runs
    under uvicorn with piped stdout the pipe buffer can fill or close,
    raising ``BrokenPipeError ([Errno 32] Broken pipe)``.
    Suppressing stdout eliminates this entirely.
    """
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def load_event_log(xes_path: str):
    """Load an XES event log file."""
    with normalized_xes_path(xes_path) as import_path:
        with _suppress_stdout():
            return xes_importer.apply(import_path)


def log_to_dataframe_preserve_all(event_log):
    """Convert event log to DataFrame using PM4Py's built-in converter to preserve all attributes."""
    with _suppress_stdout():
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

    logger.info("Loading XES file: %s", xes_path)
    event_log = load_event_log(xes_path)

    df = log_to_dataframe_preserve_all(event_log)

    if df.empty or len(df.columns) < 3:
        logger.info("Primary conversion empty, using manual method...")
        df = log_to_dataframe_manual(event_log)

    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)

    logger.info("[OK] XES converted to CSV: %s | columns=%s | events=%d",
                csv_path, list(df.columns), len(df))

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
