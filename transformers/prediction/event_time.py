import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from ..model import build_time_prediction_model
from .training_logger import EpochMetricsLogger


class EventTimePredictor:
    def __init__(self, max_len=16, d_model=64, num_heads=4, num_blocks=2, 
                 dropout_rate=0.1):
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.vocab_size = None
        self.history = None
    
    def prepare_data(self, df, test_size=0.3, val_split=0.5):
        print("Preparing data for Event Time Prediction...")
        
        required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) != len(required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {available_cols}")
        
        split_col = "__split" if "__split" in df.columns else None
        select_cols = required_cols + ([split_col] if split_col else [])
        df = df[select_cols].copy()
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            print(f"[WARN] Dropping duplicate columns: {dupes}")
            df = df.loc[:, ~df.columns.duplicated()]
        
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        
        df = self._calculate_temporal_features(df)
        
        self.label_encoder.fit(df['concept:name'])
        df['activity_encoded'] = self.label_encoder.transform(df['concept:name'])
        
        def build_samples(sub_df):
            grouped = sub_df.groupby('case:concept:name')
            sequences = []
            temporal_features = []
            next_event_times = []
            time_sequences = []

            for case_id, group in grouped:
                activities = group['activity_encoded'].values
                timestamps = group['time:timestamp'].values
                fvt1_vals = group['fvt1'].values
                fvt2_vals = group['fvt2'].values
                fvt3_vals = group['fvt3'].values
                case_start = timestamps[0]

                for i in range(1, len(activities)):
                    seq = activities[:i]
                    temp_feat = [fvt1_vals[i], fvt2_vals[i], fvt3_vals[i]]
                    time_to_next = (timestamps[i] - timestamps[i-1]).astype('timedelta64[s]').astype(float) / 86400
                    time_seq = ((timestamps[:i] - case_start).astype('timedelta64[s]').astype(float) / 86400)

                    sequences.append(seq)
                    temporal_features.append(temp_feat)
                    next_event_times.append(time_to_next)
                    time_sequences.append(time_seq)

            return sequences, np.array(temporal_features), np.array(next_event_times), time_sequences

        if split_col:
            split_values = set(df[split_col].dropna().unique().tolist())
            if not {"train", "val", "test"}.issubset(split_values):
                raise ValueError("Split column must include train, val, and test values.")

            train_df = df[df[split_col] == "train"].drop(columns=[split_col])
            val_df = df[df[split_col] == "val"].drop(columns=[split_col])
            test_df = df[df[split_col] == "test"].drop(columns=[split_col])

            seq_train, X_temp_train, y_train, time_train = build_samples(train_df)
            seq_val, X_temp_val, y_val, time_val = build_samples(val_df)
            seq_test, X_temp_test, y_test, time_test = build_samples(test_df)

            X_seq_train = keras.preprocessing.sequence.pad_sequences(
                seq_train, maxlen=self.max_len, padding='pre', value=0
            ) + 1
            X_seq_val = keras.preprocessing.sequence.pad_sequences(
                seq_val, maxlen=self.max_len, padding='pre', value=0
            ) + 1
            X_seq_test = keras.preprocessing.sequence.pad_sequences(
                seq_test, maxlen=self.max_len, padding='pre', value=0
            ) + 1
            X_time_train = keras.preprocessing.sequence.pad_sequences(
                time_train, maxlen=self.max_len, padding='pre', value=0
            )
            X_time_val = keras.preprocessing.sequence.pad_sequences(
                time_val, maxlen=self.max_len, padding='pre', value=0
            )
            X_time_test = keras.preprocessing.sequence.pad_sequences(
                time_test, maxlen=self.max_len, padding='pre', value=0
            )

            X_temp_train_scaled = self.scaler.fit_transform(X_temp_train)
            X_temp_val_scaled = self.scaler.transform(X_temp_val) if len(X_temp_val) else X_temp_val
            X_temp_test_scaled = self.scaler.transform(X_temp_test) if len(X_temp_test) else X_temp_test
        else:
            sequences, X_temp, y_event_time, time_seq = build_samples(df)
            print(f"Total training samples: {len(sequences):,}")
            print(f"Example sequence length range: {min(map(len, sequences))} to {max(map(len, sequences))}")

            X_seq = keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=self.max_len, padding='pre', value=0
            ) + 1
            X_time = keras.preprocessing.sequence.pad_sequences(
                time_seq, maxlen=self.max_len, padding='pre', value=0
            )

            X_temp_scaled = self.scaler.fit_transform(X_temp)

            X_seq_train, X_seq_temp, X_temp_train_scaled, X_temp_temp, X_time_train, X_time_temp, y_train, y_temp = train_test_split(
                X_seq, X_temp_scaled, X_time, y_event_time, test_size=test_size, random_state=42
            )
            
            X_seq_val, X_seq_test, X_temp_val_scaled, X_temp_test_scaled, X_time_val, X_time_test, y_val, y_test = train_test_split(
                X_seq_temp, X_temp_temp, X_time_temp, y_temp, test_size=val_split, random_state=42
            )

            X_temp_train_scaled = X_temp_train_scaled

        self.vocab_size = len(self.label_encoder.classes_) + 2
        
        print(f"\nSequence shape: {X_seq_train.shape}")
        print(f"Temporal features shape: {X_temp_train_scaled.shape}")
        print(f"Event time targets shape: {y_train.shape}")
        print(f"Vocabulary size: {self.vocab_size}")
        
        # ==================== NEW: Extract Timestamps for Explainability ====================
        print("\nExtracting timestamps for explainability...")
        
        def extract_timestamps_for_sequences(df_data, max_length):
            """Extract timestamp labels for each sequence"""
            all_timestamps = []
            
            for case_id, group in df_data.groupby('case:concept:name'):
                group = group.sort_values('time:timestamp').reset_index(drop=True)
                timestamps = group['time:timestamp'].values
                
                # Calculate relative time from case start
                start_time = timestamps[0]
                
                # Create prefix sequences (same as training data generation)
                for i in range(1, len(timestamps)):
                    # Get timestamps for prefix up to position i
                    prefix_timestamps = timestamps[:i]
                    
                    # Calculate days from start for each timestamp
                    timestamp_labels = []
                    for j, ts in enumerate(prefix_timestamps):
                        days_diff = (ts - start_time).astype('timedelta64[s]').astype(float) / 86400
                        if days_diff == 0:
                            timestamp_labels.append("Day 0")
                        else:
                            timestamp_labels.append(f"Day {days_diff:.1f}")
                    
                    # Pad to max_length (prepend padding like sequences)
                    if len(timestamp_labels) < max_length:
                        padded = ['[PAD]'] * (max_length - len(timestamp_labels)) + timestamp_labels
                    else:
                        padded = timestamp_labels[-max_length:]
                    
                    all_timestamps.append(padded)
            
            return all_timestamps
        
        # Extract timestamps for all data
        all_timestamps = extract_timestamps_for_sequences(df, self.max_len)
        
        # Split timestamps same way as sequences
        timestamps_train, timestamps_temp = train_test_split(
            all_timestamps, test_size=test_size, random_state=42
        )
        
        timestamps_val, timestamps_test = train_test_split(
            timestamps_temp, test_size=val_split, random_state=42
        )
        
        print(f"Extracted timestamps for {len(timestamps_test)} test samples")
        # ==================== END: Timestamp Extraction ====================
        
        print(f"\nDataset splits:")
        print(f"Train: {len(X_seq_train):,} samples")
        print(f"Validation: {len(X_seq_val):,} samples")
        print(f"Test: {len(X_seq_test):,} samples")
        
        return {
            'X_seq_train': X_seq_train, 'X_temp_train': X_temp_train_scaled, 'y_train': y_train,
            'X_seq_val': X_seq_val, 'X_temp_val': X_temp_val_scaled, 'y_val': y_val,
            'X_seq_test': X_seq_test, 'X_temp_test': X_temp_test_scaled, 'y_test': y_test,
            'X_time_train': X_time_train, 'X_time_val': X_time_val, 'X_time_test': X_time_test
        }
    
    def _calculate_temporal_features(self, df):
        def calculate_features(group):
            group = group.sort_values('time:timestamp').reset_index(drop=True)
            
            group['fvt1'] = group['time:timestamp'].diff().dt.total_seconds() / 86400
            group['fvt1'].fillna(0, inplace=True)
            
            group['fvt2'] = (group['time:timestamp'] - group['time:timestamp'].shift(2)).dt.total_seconds() / 86400
            group['fvt2'].fillna(0, inplace=True)
            
            group['fvt3'] = (group['time:timestamp'] - group['time:timestamp'].iloc[0]).dt.total_seconds() / 86400
            
            return group
        
        df = df.groupby('case:concept:name', group_keys=False).apply(calculate_features)
        print("Temporal features created.")
        return df
    
    def build_model(self, use_timestep_explainability=True):
        """
        Build Event Time Prediction Model.
        
        Args:
            use_timestep_explainability: If True (default), enables timestep-level 
                                        explanations for SHAP/LIME
        """
        print("\nBuilding Event Time Prediction Model...")
        print(f"Timestep explainability: {'ENABLED' if use_timestep_explainability else 'DISABLED'}")
        
        self.model = build_time_prediction_model(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            use_timestep_explainability=use_timestep_explainability
        )

        # Handle compilation based on model type
        if len(self.model.outputs) > 1:
            print("Model type: Timestep-explainable (2 outputs)")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=['mae', None],
                metrics={'time_output': ['mae']}
            )
        else:
            print("Model type: Original (1 output)")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mae',
                metrics=['mae']
            )
        
        print("\nModel Summary:")
        self.model.summary()
    
    def train(self, data, epochs=50, batch_size=128, patience=10):
        print(f"\nTraining model for {epochs} epochs...")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        epoch_logger = EpochMetricsLogger()

        if len(self.model.outputs) > 1:
            self.history = self.model.fit(
                [data['X_seq_train'], data['X_temp_train']], 
                [data['y_train'], None],
                validation_data=(
                    [data['X_seq_val'], data['X_temp_val']], 
                    [data['y_val'], None]
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[epoch_logger, early_stopping],
                verbose=0
            )
        else:
            self.history = self.model.fit(
                [data['X_seq_train'], data['X_temp_train']], 
                data['y_train'],
                validation_data=([data['X_seq_val'], data['X_temp_val']], data['y_val']),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[epoch_logger, early_stopping],
                verbose=0
            )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self, data):
        print("\nEvaluating on test set...")
        
        if len(self.model.outputs) > 1:
            results = self.model.evaluate(
                [data['X_seq_test'], data['X_temp_test']], 
                [data['y_test'], None],
                verbose=0
            )
            test_loss = results[1] if len(results) > 1 else results[0]
            test_mae = results[2] if len(results) > 2 else results[-1]
        else:
            test_loss, test_mae = self.model.evaluate(
                [data['X_seq_test'], data['X_temp_test']], 
                data['y_test'], 
                verbose=0
            )
        
        print(f"Test MAE: {test_mae:.4f} days")
        print(f"Test Loss: {test_loss:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_mae': test_mae
        }
    
    def predict(self, data):
        print("\nGenerating predictions...")
        
        outputs = self.model.predict(
            [data['X_seq_test'], data['X_temp_test']], 
            verbose=0
        )
        
        if isinstance(outputs, list) and len(outputs) > 1:
            y_pred = outputs[0].flatten()
            timestep_preds = outputs[1]
            print(f"Predictions generated for {len(data['X_seq_test']):,} test samples.")
            print(f"  - Final predictions shape: {y_pred.shape}")
            print(f"  - Timestep predictions shape: {timestep_preds.shape}")
            return y_pred, timestep_preds
        else:
            y_pred = outputs.flatten() if hasattr(outputs, 'flatten') else outputs
            print(f"Predictions generated for {len(data['X_seq_test']):,} test samples.")
            return y_pred, None
    
    def save_results(self, data, y_pred, output_dir):
        print("\nSaving results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        
        results = pd.DataFrame({
            'actual_event_time_days': data['y_test'],
            'predicted_event_time_days': y_pred,
            'absolute_error_days': np.abs(data['y_test'] - y_pred)
        })
        
        output_path = os.path.join(output_dir, "event_time_predictions.csv")
        results.to_csv(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
    
    def plot_predictions(self, data, y_pred, output_dir):
        print("\nPlotting predictions...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(data['y_test'], y_pred, alpha=0.5, s=10)
        plt.plot([data['y_test'].min(), data['y_test'].max()], 
                 [data['y_test'].min(), data['y_test'].max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Event Time (days)', fontsize=12)
        plt.ylabel('Predicted Event Time (days)', fontsize=12)
        
        mae = np.mean(np.abs(data['y_test'] - y_pred))
        plt.title(f'Event Time Prediction\nMAE: {mae:.4f} days', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "event_time_predictions_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions plot saved to: {output_path}")
    
    def plot_training_history(self, output_dir):
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        print("\nPlotting training history...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if 'loss' in self.history.history:
            train_loss_key = 'loss'
            val_loss_key = 'val_loss'
        elif 'time_output_loss' in self.history.history:
            train_loss_key = 'time_output_loss'
            val_loss_key = 'val_time_output_loss'
        else:
            print("Could not find loss keys in history")
            return
        
        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, len(self.history.history[train_loss_key]) + 1)
        epoch_xlim = (0.5, 1.5) if len(epochs) == 1 else (1, int(epochs[-1]))
        plt.plot(epochs, self.history.history[train_loss_key], label="Training Loss", linewidth=2)
        plt.plot(epochs, self.history.history[val_loss_key], label="Validation Loss", linewidth=2)
        plt.title("Event Time Prediction - Loss Over Time", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MAE)", fontsize=12)
        plt.xlim(*epoch_xlim)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "event_time_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Training history plot saved to: {output_path}")
        
        print("\nTraining metrics:")
        print(f"Final training loss: {self.history.history[train_loss_key][-1]:.4f}")
        print(f"Final validation loss: {self.history.history[val_loss_key][-1]:.4f}")
        print(f"Best validation loss: {min(self.history.history[val_loss_key]):.4f}")
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "event_time_transformer.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
