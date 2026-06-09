import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import glob
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
np.random.seed(42)

EXPLAINABILITY_AVAILABLE = True
try:
    from explainability.transformers import run_transformer_explainability
    from explainability.gnns import run_gnn_explainability
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    print("[WARNING] Warning: Explainability modules not available.")

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] Warning: TensorFlow not available. Transformer models will not work.")

try:
    import torch
    torch.manual_seed(42)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[WARNING] Warning: PyTorch not available. GNN models will not work.")

if TENSORFLOW_AVAILABLE:
    from transformers.prediction.next_activity import NextActivityPredictor
    from transformers.prediction.event_time import EventTimePredictor
    from transformers.prediction.remaining_time import RemainingTimePredictor

if PYTORCH_AVAILABLE:
    from gnns.prediction.gnn_predictor import GNNPredictor


from conv_and_viz.xes_to_csv import load_event_log, log_to_dataframe_preserve_all
from conv_and_viz.preprocessor_csv import preprocess_event_log

DATASET_DIRECTORY = "BPI_Models/BPI_logs_preprocessed_csv"
BASE_OUTPUT_DIR = "results"


def print_banner():
    print("\n" + "="*70)
    print(" "*15 + "PREDICTIVE PROCESS MONITORING")
    print("="*70 + "\n")


def get_user_choice(prompt, options):
    print(f"\n{prompt}")
    for num, desc in options.items():
        print(f"  {num}. {desc}")
    while True:
        try:
            choice = int(input("\nEnter your choice: "))
            if choice in options:
                return choice
            else:
                print(f"Invalid choice. Please select from {list(options.keys())}")
        except ValueError:
            print("Please enter a valid number.")


def get_yes_no(prompt):
    while True:
        response = input(f"\n{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def get_file_format_choice():
    """Let user select between XES or CSV file format."""
    print("\n" + "="*70)
    print("FILE FORMAT SELECTION")
    print("="*70)
    format_options = {
        1: "XES (eXtensible Event Stream) - Standard process mining format",
        2: "CSV (Comma-Separated Values) - Tabular data format"
    }
    choice = get_user_choice("Select input file format:", format_options)
    return "xes" if choice == 1 else "csv"


def get_dataset_files(file_format):
    """Get dataset files based on selected format."""
    print("\n" + "-"*70)
    print("DATASET SELECTION")
    print("-"*70)
    
    if file_format == "xes":
        search_dir = "BPI_Models/BPI_logs_xes"
        file_ext = "*.xes"
    else:
        
        if os.path.exists(DATASET_DIRECTORY) and glob.glob(os.path.join(DATASET_DIRECTORY, "*.csv")):
            search_dir = DATASET_DIRECTORY
        else:
            search_dir = "BPI_Models/BPI_logs_csv"
        file_ext = "*.csv"
    
    if not os.path.exists(search_dir):
        print(f"[X] Directory not found: {search_dir}")
        print("Please create the directory and add your files")
        sys.exit(1)
    
    file_pattern = os.path.join(search_dir, file_ext)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"[X] No {file_format.upper()} files found in: {search_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(files)} {file_format.upper()} file(s) in {search_dir}:\n")
    for idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {idx}. {filename} ({filesize:.2f} MB)")
    
    print("\n" + "-"*70)
    
    dataset_options = {idx: os.path.basename(files[idx-1]) for idx in range(1, len(files)+1)}
    choice = get_user_choice("Select dataset:", dataset_options)
    
    selected_file = files[choice - 1]
    print(f"\n[OK] Selected: {os.path.basename(selected_file)}")
    return selected_file


def convert_xes_to_csv(xes_path, output_dir):
    """Convert XES file to CSV format."""
    print(f"\nConverting XES to CSV...")
    event_log = load_event_log(xes_path)
    df = log_to_dataframe_preserve_all(event_log)
    
    if df.empty or len(df.columns) < 3:
        print("[WARNING] Standard conversion failed, using manual method...")
        
        data = []
        for trace in event_log:
            trace_attrs = dict(trace.attributes)
            for event in trace:
                row = trace_attrs.copy()
                row.update(dict(event))
                data.append(row)
        df = pd.DataFrame(data)
    
    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"[OK] XES converted to CSV: {csv_path}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Events: {len(df):,}")
    
    return csv_path, df, list(df.columns)


def process_input_file(input_path, file_format):
    """
    Process input file: XES→CSV conversion and/or CSV preprocessing.
    Returns path to preprocessed CSV.
    """
    print("\n" + "="*70)
    print("FILE PROCESSING PIPELINE")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    
    if file_format == "xes":
        print("\n[Step 1/2] Converting XES to CSV...")
        temp_csv_dir = os.path.join("temp_processing", timestamp)
        os.makedirs(temp_csv_dir, exist_ok=True)
        
        csv_path, _, _ = convert_xes_to_csv(input_path, temp_csv_dir)
        print(f"✓ XES converted to CSV")
    else:
        print("\n[Step 1/2] CSV file detected, skipping XES conversion...")
        csv_path = input_path
        print(f"✓ Using CSV file directly")
    
    
    print("\n[Step 2/2] Preprocessing CSV (cleaning, deduplication, missing values)...")
    preprocessed_dir = os.path.join("temp_processing", f"preprocessed_{timestamp}")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    preprocessed_path = os.path.join(preprocessed_dir, f"{base_name}_preprocessed.csv")
    
    try:
        df = preprocess_event_log(csv_path, preprocessed_path)
        print(f"✓ Preprocessing complete")
    except Exception as e:
        print(f"[WARNING] Preprocessing failed: {e}")
        print("Using raw CSV file...")
        preprocessed_path = csv_path
    
    print("\n" + "="*70)
    print("FILE PROCESSING COMPLETE")
    print("="*70)
    print(f"✓ Preprocessed dataset: {preprocessed_path}")
    print("="*70)
    
    return preprocessed_path


def manual_column_mapping(df):
    """Interactive column mapping similar to DISCO - user selects each column manually."""
    print("\n" + "="*70)
    print("MANUAL COLUMN MAPPING (DISCO-style)")
    print("="*70)
    print("\nAvailable columns in your dataset:")
    for idx, col in enumerate(df.columns, 1):
        unique_count = df[col].nunique()
        dtype = df[col].dtype
        sample_val = str(df[col].iloc[0])[:30]
        print(f"  {idx}. {col} ({dtype}, {unique_count} unique, e.g., '{sample_val}')")
    
    column_mapping = {}
    
    print("\n" + "-"*70)
    print("Map columns to required attributes:")
    print("-"*70)
    
    
    while True:
        print(f"\n[1/3] Select CASE ID column (identifies process instances):")
        print("Available columns:")
        for idx, col in enumerate(df.columns, 1):
            print(f"  {idx}. {col}")
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(df.columns):
                case_col = df.columns[choice - 1]
                print(f"✓ Selected: {case_col}")
                if case_col != 'CaseID':
                    column_mapping[case_col] = 'CaseID'
                break
            else:
                print(f"Invalid choice. Please select 1-{len(df.columns)}")
        except ValueError:
            print("Please enter a valid number.")
    
    
    while True:
        print(f"\n[2/3] Select ACTIVITY column (contains activity/event names):")
        print("Available columns:")
        for idx, col in enumerate(df.columns, 1):
            unique = df[col].nunique()
            print(f"  {idx}. {col} ({unique} unique activities)")
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(df.columns):
                activity_col = df.columns[choice - 1]
                print(f"✓ Selected: {activity_col} ({df[activity_col].nunique()} unique activities)")
                if activity_col != 'Activity':
                    column_mapping[activity_col] = 'Activity'
                break
            else:
                print(f"Invalid choice. Please select 1-{len(df.columns)}")
        except ValueError:
            print("Please enter a valid number.")
    
    
    while True:
        print(f"\n[3/3] Select TIMESTAMP column (event time):")
        print("Available columns:")
        for idx, col in enumerate(df.columns, 1):
            sample_val = str(df[col].iloc[0])[:40]
            print(f"  {idx}. {col} (e.g., {sample_val})")
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(df.columns):
                timestamp_col = df.columns[choice - 1]
                print(f"✓ Selected: {timestamp_col}")
                if timestamp_col != 'Timestamp':
                    column_mapping[timestamp_col] = 'Timestamp'
                break
            else:
                print(f"Invalid choice. Please select 1-{len(df.columns)}")
        except ValueError:
            print("Please enter a valid number.")
    
    
    include_resource = get_yes_no("\nDo you want to include a RESOURCE column? (optional)")
    if include_resource:
        while True:
            print(f"\nSelect RESOURCE column (who/what performed the activity):")
            print("Available columns:")
            for idx, col in enumerate(df.columns, 1):
                print(f"  {idx}. {col}")
            try:
                choice = int(input("\nEnter number: "))
                if 1 <= choice <= len(df.columns):
                    resource_col = df.columns[choice - 1]
                    print(f"✓ Selected: {resource_col}")
                    if resource_col != 'Resource':
                        column_mapping[resource_col] = 'Resource'
                    break
                else:
                    print(f"Invalid choice. Please select 1-{len(df.columns)}")
            except ValueError:
                print("Please enter a valid number.")
    
    if column_mapping:
        print("\n" + "="*70)
        print("COLUMN MAPPING SUMMARY")
        print("="*70)
        for old, new in column_mapping.items():
            print(f"{new.upper()}: '{old}' -> '{new}'")
        print("="*70)
        df = df.rename(columns=column_mapping)
    
    return df, column_mapping


def detect_and_standardize_columns(df, verbose=True):
    """Auto-detection of standard column names."""
    column_mapping = {}
    columns_to_drop = []
    
    case_patterns = ['case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    activity_patterns = ['concept:name', 'Activity', 'activity', 'Action', 'event', 'Event', 'task', 'Task']
    timestamp_patterns = ['time:timestamp', 'Timestamp', 'timestamp', 'time', 'Time', 'start_time', 'StartTime', 'complete_time', 'CompleteTime']
    resource_patterns = ['org:resource', 'Resource', 'resource', 'user', 'User', 'org:role', 'org:group', 'role', 'Role', 'actor', 'Actor']
    
    for col in df.columns:
        if col in case_patterns and col != 'CaseID':
            column_mapping[col] = 'CaseID'
            break
    
    for col in df.columns:
        if col in activity_patterns and col != 'Activity':
            if 'Activity' in df.columns and col != 'Activity':
                columns_to_drop.append('Activity')
            column_mapping[col] = 'Activity'
            break
    
    for col in df.columns:
        if col in timestamp_patterns and col != 'Timestamp':
            column_mapping[col] = 'Timestamp'
            break
    
    for col in df.columns:
        if col in resource_patterns and col != 'Resource':
            column_mapping[col] = 'Resource'
            break
    
    if columns_to_drop:
        if verbose:
            print(f"Dropping conflicting columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    if verbose and column_mapping:
        print("="*70)
        print("AUTO-DETECTION RESULTS")
        print("="*70)
        for old, new in column_mapping.items():
            print(f"{new.upper()}: '{old}' -> '{new}'")
        print("="*70)
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    required = ['CaseID', 'Activity', 'Timestamp']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")
    
    return df, column_mapping, column_mapping.keys()


def get_file_initials(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) >= 2:
        initials = ''.join([p[0].upper() for p in parts[:3] if p and len(p) > 0])
        return initials[:5]
    else:
        return name[:5].upper()


def create_output_directory(dataset_path, task_name, model_type, explainability_method):
    initials = get_file_initials(dataset_path)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    task_short = task_name.replace(" ", "_").lower()
    
    model_name = "GNN" if model_type == "gnn" else "Transformer"
    
    if explainability_method is None:
        explainability_part = "without_explainability"
    elif explainability_method == "all":
        explainability_part = "shap_lime" if model_type != "gnn" else "gradient_lime"
    else:
        explainability_part = explainability_method
    
    folder_name = f"{model_name}_{task_short}_{explainability_part}_{timestamp}"
    output_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[OK] Output directory created: {output_dir}")
    
    info_file = os.path.join(output_dir, "dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Full Path: {os.path.abspath(dataset_path)}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Model Type: {model_name}\n")
        f.write(f"Explainability: {explainability_part if explainability_method else 'None'}\n")
        f.write(f"Timestamp: {timestamp}\n")
    return output_dir


def get_data_split():
    print("\n" + "-"*70)
    print("DATA SPLIT CONFIGURATION")
    print("-"*70)
    split_options = {
        1: "70-15-15 (Train-Val-Test)",
        2: "80-10-10 (Train-Val-Test)",
        3: "60-20-20 (Train-Val-Test)",
        4: "Custom split"
    }
    choice = get_user_choice("Select data split:", split_options)
    if choice == 1:
        return 0.3, 0.5
    elif choice == 2:
        return 0.2, 0.5
    elif choice == 3:
        return 0.4, 0.5
    else:
        while True:
            try:
                train = float(input("Enter training set percentage (e.g., 70 for 70%): "))
                val = float(input("Enter validation set percentage (e.g., 15 for 15%): "))
                test = float(input("Enter test set percentage (e.g., 15 for 15%): "))

                if train + val + test == 100 and all([train > 0, val > 0, test > 0]):
                    test_size = (val + test) / 100
                    val_split = val / (val + test)
                    return test_size, val_split
                else:
                    print("Percentages must sum to 100 and all be positive.")
            except ValueError:
                print("Please enter valid numbers.")


def get_explainability_choice(model_type='transformer'):
    if not EXPLAINABILITY_AVAILABLE:
        return None
    
    print("\n" + "-"*70)
    print("EXPLAINABILITY CONFIGURATION")
    print("-"*70)
    print("\nExplainability helps understand model predictions by:")
    print("  - Identifying important features")
    print("  - Visualizing decision-making process")
    print("  - Providing interpretable insights")
    print("\nNote: Explainability analysis may take additional time")
    
    if model_type == 'gnn':
        explainability_options = {
            1: "Gradient-Based Attribution",
            2: "GraphLIME",
            3: "All methods (Gradient + GraphLIME)",
            4: "Skip Explainability"
        }
    else:
        explainability_options = {
            1: "SHAP",
            2: "LIME",
            3: "All methods (SHAP + LIME)",
            4: "Skip Explainability"
        }
    
    choice = get_user_choice("Select explainability method:", explainability_options)
    
    if model_type == 'gnn':
        if choice == 1:
            return "gradient"
        elif choice == 2:
            return "lime"
        elif choice == 3:
            return "all"
        else:
            return None
    else:
        if choice == 1:
            return "shap"
        elif choice == 2:
            return "lime"
        elif choice == 3:
            return "all"
        else:
            return None


def get_gnn_config():
    print("\n" + "-"*70)
    print("GNN MODEL CONFIGURATION")
    print("-"*70)
    use_default = get_yes_no("Use default configuration?")
    if use_default:
        print("\nUsing default configuration:")
        config = {
            'hidden': 64,
            'dropout_rate': 0.1,
            'lr': 4e-4,
            'epochs': 5,
            'batch_size': 64,
            'patience': 10
        }
    else:
        print("\nEnter custom configuration:")
        config = {}
        try:
            config['hidden'] = int(input("  Hidden channels [64]: ") or 64)
            config['dropout_rate'] = float(input("  Dropout rate [0.1]: ") or 0.1)
            config['lr'] = float(input("  Learning rate [4e-4]: ") or 4e-4)
            config['epochs'] = int(input("  Number of epochs [5]: ") or 5)
            config['batch_size'] = int(input("  Batch size [64]: ") or 64)
            config['patience'] = int(input("  Early stopping patience [10]: ") or 10)
        except ValueError:
            print("Invalid input. Using default configuration.")
            return get_gnn_config()
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return config


def get_model_config():
    print("\n" + "-"*70)
    print("MODEL CONFIGURATION")
    print("-"*70)
    use_default = get_yes_no("Use default configuration?")
    if use_default:
        print("\nUsing default configuration:")
        config = {
            'max_len': 16,
            'd_model': 64,
            'num_heads': 4,
            'num_blocks': 2,
            'dropout_rate': 0.1,
            'epochs': 5,
            'batch_size': 128,
            'patience': 10
        }
    else:
        print("\nEnter custom configuration:")
        config = {}
        try:
            config['max_len'] = int(input("  Max sequence length [16]: ") or 16)
            config['d_model'] = int(input("  Model dimension [64]: ") or 64)
            config['num_heads'] = int(input("  Number of attention heads [4]: ") or 4)
            config['num_blocks'] = int(input("  Number of transformer blocks [2]: ") or 2)
            config['dropout_rate'] = float(input("  Dropout rate [0.1]: ") or 0.1)
            config['epochs'] = int(input("  Number of epochs [5]: ") or 5)
            config['batch_size'] = int(input("  Batch size [128]: ") or 128)
            config['patience'] = int(input("  Early stopping patience [10]: ") or 10)
        except ValueError:
            print("Invalid input. Using default configuration.")
            return get_model_config()

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


def run_next_activity_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method):
    print("\n" + "="*70)
    print("NEXT ACTIVITY PREDICTION")
    print("="*70)
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    
    required_cols = {'CaseID', 'Activity', 'Timestamp'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    
    df = df.rename(columns={
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    
    print(f"Dataset loaded: {len(df):,} events")

    predictor = NextActivityPredictor(
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
    y_pred, y_pred_probs = predictor.predict(data)
    predictor.save_results(data, y_pred, y_pred_probs, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    
    if explainability_method and EXPLAINABILITY_AVAILABLE:
        print("\nRunning explainability analysis...")
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='activity',
            num_samples=20,
            methods=explainability_method,
            label_encoder=predictor.label_encoder,
            scaler=None
        )
    
    print("\n" + "="*70)
    print("NEXT ACTIVITY PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test Accuracy':<30} {metrics['test_accuracy']*100:>19.2f}%")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_test']):>20,}")
    print("-"*70)
    print(f"\n[OK] All results saved to: {output_dir}")
    print("="*70)


def run_event_time_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method):
    print("\n" + "="*70)
    print("EVENT TIME PREDICTION")
    print("="*70)
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    
    required_cols = {'CaseID', 'Activity', 'Timestamp'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    
    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    
    print(f"Dataset loaded: {len(df):,} events")
    
    predictor = EventTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )
    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    
    predictor.build_model()  # Uses timestep explainability by default
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    metrics = predictor.evaluate(data)
    
    # predict() now returns a tuple: (y_pred, timestep_preds)
    y_pred, timestep_preds = predictor.predict(data)
    
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    
    if explainability_method and EXPLAINABILITY_AVAILABLE:
        print("\nRunning explainability analysis...")
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=20,
            methods=explainability_method,
            scaler=predictor.scaler,
            label_encoder=predictor.label_encoder,
            timestamps=data.get('X_time_test')
        )
    
    print("\n" + "="*70)
    print("EVENT TIME PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test MAE (Mean Absolute Error)':<30} {metrics['test_mae']:>17.4f} days")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_seq_test']):>20,}")
    print("-"*70)
    print(f"\n[OK] All results saved to: {output_dir}")
    print("="*70)


def run_remaining_time_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method):
    print("\n" + "="*70)
    print("REMAINING TIME PREDICTION")
    print("="*70)
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    
    required_cols = {'CaseID', 'Activity', 'Timestamp'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    
    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    
    print(f"Dataset loaded: {len(df):,} events")
    
    predictor = RemainingTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )
    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    
    predictor.build_model()  # Uses timestep explainability by default
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    metrics = predictor.evaluate(data)
    
    # predict() now returns a tuple: (y_pred, timestep_preds)
    y_pred, timestep_preds = predictor.predict(data)
    
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    
    if explainability_method and EXPLAINABILITY_AVAILABLE:
        print("\nRunning explainability analysis...")
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=20,
            methods=explainability_method,
            label_encoder=predictor.label_encoder,
            scaler=predictor.scaler,
            timestamps=data.get('X_time_test')
        )
    
    print("\n" + "="*70)
    print("REMAINING TIME PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test MAE (Mean Absolute Error)':<30} {metrics['test_mae']:>17.4f} days")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_seq_test']):>20,}")
    print("-"*70)
    print(f"\n[OK] All results saved to: {output_dir}")
    print("="*70)


def run_gnn_unified_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method, task='unified'):
    print("\n" + "="*70)
    if task == 'unified':
        print("GNN UNIFIED PREDICTION")
        print("All three tasks: Activity + Event Time + Remaining Time")
    else:
        task_names = {
            'next_activity': 'NEXT ACTIVITY PREDICTION',
            'event_time': 'EVENT TIME PREDICTION',
            'remaining_time': 'REMAINING TIME PREDICTION'
        }
        print(f"GNN {task_names.get(task, 'PREDICTION')}")
    print("="*70)
    
    if not PYTORCH_AVAILABLE:
        print("\n[X] PyTorch not available. Please install PyTorch and PyTorch Geometric.")
        return
    
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    
    required_cols = {'CaseID', 'Activity', 'Timestamp'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)
    print(f"Dataset loaded: {len(df):,} events")

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
        loss_weights=loss_weights
    )

    data = predictor.prepare_data(
        df, 
        test_size=test_size, 
        val_split=val_split
    )

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
    predictor.save_model(output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_results(metrics, output_dir)
    
    if explainability_method and EXPLAINABILITY_AVAILABLE:
     print("\nRunning explainability analysis...")
    explainability_dir = os.path.join(output_dir, 'explainability')
    
    
    vocabularies = getattr(predictor, 'vocabs', data.get('vocabs', {}))
    
    # Determine which task to explain based on gnn_task
    if task == 'next_activity':
        explain_tasks = ['activity']
    elif task == 'event_time':
        explain_tasks = ['event_time']
    elif task == 'remaining_time':
        explain_tasks = ['remaining_time']
    else:  # unified
        explain_tasks = ['activity', 'event_time', 'remaining_time']
    
    run_gnn_explainability(
        model=predictor.model,
        data=data,
        output_dir=explainability_dir,
        device=predictor.device,
        vocabularies=vocabularies,
        num_samples=10,
        methods=explainability_method,
        tasks=explain_tasks,
        global_sample_percent=config.get('global_sample_percent', 0.01)
    )
    
    print("\n" + "="*70)
    if task == 'unified':
        print("GNN UNIFIED PREDICTION - FINAL RESULTS")
        print("="*70)
        print(f"\n{'Metric':<35} {'Value':>20}")
        print("-"*70)
        print(f"{'Next Activity Accuracy':<35} {metrics['accuracy']*100:>19.2f}%")
        print(f"{'Event Time MAE':<35} {metrics['mae_time']:>20.4f}")
        print(f"{'Remaining Time MAE':<35} {metrics['mae_rem']:>20.4f}")
        print(f"{'Total Loss':<35} {metrics['loss']:>20.4f}")
    elif task == 'next_activity':
        print("GNN NEXT ACTIVITY PREDICTION - FINAL RESULTS")
        print("="*70)
        print(f"\n{'Metric':<35} {'Value':>20}")
        print("-"*70)
        print(f"{'Test Accuracy':<35} {metrics['accuracy']*100:>19.2f}%")
        print(f"{'Total Loss':<35} {metrics['loss']:>20.4f}")
    elif task == 'event_time':
        print("GNN EVENT TIME PREDICTION - FINAL RESULTS")
        print("="*70)
        print(f"\n{'Metric':<35} {'Value':>20}")
        print("-"*70)
        print(f"{'Event Time MAE':<35} {metrics['mae_time']:>20.4f}")
        print(f"{'Total Loss':<35} {metrics['loss']:>20.4f}")
    elif task == 'remaining_time':
        print("GNN REMAINING TIME PREDICTION - FINAL RESULTS")
        print("="*70)
        print(f"\n{'Metric':<35} {'Value':>20}")
        print("-"*70)
        print(f"{'Remaining Time MAE':<35} {metrics['mae_rem']:>20.4f}")
        print(f"{'Total Loss':<35} {metrics['loss']:>20.4f}")
    print("-"*70)
    print(f"\n[OK] All results saved to: {output_dir}")
    print("="*70)


def main():
    print_banner()

    # Step 1: Select file format (XES or CSV)
    file_format = get_file_format_choice()
    
    # Step 2: Select dataset file
    raw_file_path = get_dataset_files(file_format)
    
    # Step 3: Process file (XES→CSV conversion and/or preprocessing)
    preprocessed_path = process_input_file(raw_file_path, file_format)
    
    # Step 4: Column Mapping (Auto-Detection or Manual) - AFTER PREPROCESSING
    print("\n" + "="*70)
    print("COLUMN MAPPING")
    print("="*70)
    print("\nLoading preprocessed dataset for column mapping...")
    df_preprocessed = pd.read_csv(preprocessed_path)
    
    print("\n" + "-"*70)
    print("COLUMN MAPPING OPTIONS")
    print("-"*70)
    mapping_options = {
        1: "Auto-Detection (automatic column detection)",
        2: "Manual Mapping (DISCO-style: select each column yourself)"
    }
    mapping_choice = get_user_choice("Select column mapping method:", mapping_options)
    
    if mapping_choice == 1:
        print("\n[Auto-Detection Mode]")
        try:
            df_mapped, column_mapping, detected = detect_and_standardize_columns(df_preprocessed, verbose=True)
        except ValueError as e:
            print(f"\n[X] Auto-detection failed: {e}")
            print("Falling back to Manual Mapping...")
            df_mapped, column_mapping = manual_column_mapping(df_preprocessed)
    else:
        print("\n[Manual Mapping Mode]")
        df_mapped, column_mapping = manual_column_mapping(df_preprocessed)
    
    # Verify required columns
    required_cols = {'CaseID', 'Activity', 'Timestamp'}
    if not required_cols.issubset(df_mapped.columns):
        missing = required_cols - set(df_mapped.columns)
        raise ValueError(f"Missing required columns after mapping: {missing}")
    
    # Save mapped dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join("temp_processing", f"final_{timestamp}")
    os.makedirs(final_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(raw_file_path))[0]
    final_dataset_path = os.path.join(final_dir, f"{base_name}_final.csv")
    df_mapped.to_csv(final_dataset_path, index=False)
    
    print("\n" + "="*70)
    print("COLUMN MAPPING COMPLETE")
    print("="*70)
    print(f"✓ Final dataset ready: {final_dataset_path}")
    print(f"✓ Columns: {list(df_mapped.columns)}")
    print("="*70)
    
    # Step 5: Select model type
    model_type_options = {
        1: "Transformer",
        2: "GNN (Graph Neural Network)"
    }
    model_type = get_user_choice("Select model type:", model_type_options)
    
    if model_type == 1 and not TENSORFLOW_AVAILABLE:
        print("\n" + "="*70)
        print("[X] TensorFlow not available")
        print("="*70)
        print("\nPlease install TensorFlow to use Transformer models:")
        print("  pip install tensorflow")
        sys.exit(1)
    if model_type == 2 and not PYTORCH_AVAILABLE:
        print("\n" + "="*70)
        print("[X] PyTorch not available")
        print("="*70)
        print("\nPlease install PyTorch and PyTorch Geometric to use GNN models:")
        print("  pip install torch torch-geometric")
        sys.exit(1)

    # Step 6: Select task
    if model_type == 2:
        print("\n" + "-"*70)
        print("GNN Model: Task Selection")
        print("-"*70)
        task_options = {
            1: "Next Activity Prediction",
            2: "Event Time Prediction",
            3: "Remaining Time Prediction",
            4: "All Tasks (Unified Prediction)"
        }
        task = get_user_choice("Select prediction task:", task_options)
        if task == 4:
            task_name = "GNN Unified Prediction"
            gnn_task = "unified"
        else:
            task_name = task_options[task]
            task_mapping = {
                1: "next_activity",
                2: "event_time",
                3: "remaining_time"
            }
            gnn_task = task_mapping[task]
        run_gnn = True
    else:
        task_options = {
            1: "Next Activity Prediction",
            2: "Event Time Prediction",
            3: "Remaining Time Prediction"
        }
        task = get_user_choice("Select prediction task:", task_options)
        task_name = task_options[task]
        run_gnn = False

    # Step 7: Configure explainability
    model_type_name = "gnn" if run_gnn else "transformer"
    explainability_method = get_explainability_choice(model_type=model_type_name)
    
    # Step 8: Create output directory
    output_dir = create_output_directory(final_dataset_path, task_name, model_type_name, explainability_method)
    
    # Step 9: Configure data split
    test_size, val_split = get_data_split()
    
    # Step 10: Configure model
    if run_gnn:
        config = get_gnn_config()
    else:
        config = get_model_config()
    
    # Save configuration
    config_file = os.path.join(output_dir, "configuration.txt")
    with open(config_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Type: {'GNN' if run_gnn else 'Transformer'}\n")
        f.write(f"Original File: {os.path.basename(raw_file_path)}\n")
        f.write(f"File Format: {file_format.upper()}\n")
        f.write(f"Dataset: {os.path.basename(final_dataset_path)}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Test Size: {test_size*100:.1f}%\n")
        f.write(f"Val Split: {val_split*100:.1f}%\n")
        
        if explainability_method:
            if explainability_method == 'all':
                exp_text = "All methods"
            else:
                exp_text = explainability_method.upper()
        else:
            exp_text = "Disabled"
        f.write(f"Explainability: {exp_text}\n\n")
        
        f.write("Column Mapping:\n")
        for old, new in column_mapping.items():
            f.write(f"  {old} -> {new}\n")
        f.write("\n")
        
        f.write("Model Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*50 + "\n")
    
    print(f"\n[OK] Configuration saved to: {config_file}")

    # Step 11: Train model
    try:
        if run_gnn:
            run_gnn_unified_prediction(final_dataset_path, output_dir, test_size, val_split, config, explainability_method, gnn_task)
        else:
            if task == 1:
                run_next_activity_prediction(final_dataset_path, output_dir, test_size, val_split, config, explainability_method)
            elif task == 2:
                run_event_time_prediction(final_dataset_path, output_dir, test_size, val_split, config, explainability_method)
            elif task == 3:
                run_remaining_time_prediction(final_dataset_path, output_dir, test_size, val_split, config, explainability_method)
    except Exception as e:
        print(f"\n[X] Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        error_file = os.path.join(output_dir, "error_log.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("ERROR LOG\n")
            f.write("="*50 + "\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        print(f"\n[X] Error log saved to: {error_file}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Thank you for using Predictive Process Monitoring!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
