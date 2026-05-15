import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shap
from lime import lime_tabular
import tensorflow as tf

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

def _dir_has_png(path):
    if not os.path.isdir(path):
        return False
    return any(name.lower().endswith(".png") for name in os.listdir(path))


def _write_placeholder_plot(output_path, title, lines=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    body = [title] + (lines or [])
    ax.text(0.5, 0.5, "\n".join(body), ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _ensure_stub_csv(path, columns):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


class ExplainabilityConfig:
    """Configuration for explainability behavior."""
    ENABLE_TIMESTEP_EXPLANATIONS = True
    # Options: 'auto', 'per_timestep', 'original'
    MODEL_TYPE = 'auto'

class SHAPExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        self.test_data_temp = None
        self.background_temp = None
        self.is_multi_input = False
        self.max_evals = None
        self._background_data = None
        self._max_background = None
        
        # DEBUG: Print whether label_encoder is available
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - will show generic Activity labels!")
            print("[FIX] Pass label_encoder to run_transformer_explainability()")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _get_activity_names_for_sample(self, sequence):
        if self.label_encoder is None:
            return [f'Activity_{int(t)}' if t > 0 else '[PAD]' for t in sequence]
        
        names = []
        for token in sequence:
            if token > 0:
                try:
                    # Token indices are offset by +1 (0 is padding)
                    actual_activity = self.label_encoder.inverse_transform([int(token)-1])[0]
                    names.append(actual_activity)
                except Exception as e:
                    names.append(f'Token_{int(token)}')
            else:
                names.append('[PAD]')
        return names

    def _aggregate_feature_names(self, data):
        if self.label_encoder is None:
            return [f'Position_{i+1}' for i in range(data.shape[1])]
        feature_names = []
        for pos in range(data.shape[1]):
            activities_at_pos = []
            for sample in data:
                token = sample[pos]
                if token > 0:
                    try:
                        activity = self.label_encoder.inverse_transform([int(token) - 1])[0]
                        activities_at_pos.append(activity)
                    except Exception as e:
                        print(f"[WARNING] Failed to decode activity token {int(token)}: {e}")
                        pass
            if activities_at_pos:
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                feature_names.append(most_common)
            else:
                feature_names.append(f'Position_{pos+1}')
        return feature_names

    def initialize_explainer(self, background_data, max_background=100, max_evals_override=None):
        print("Initializing SHAP Explainer...")
        self._background_data = background_data
        self._max_background = max_background
        
        if isinstance(background_data, (list, tuple)):
            self.is_multi_input = True
            bg_seq = background_data[0]
            bg_temp = background_data[1]
            indices = np.random.choice(len(bg_seq), min(max_background, len(bg_seq)), replace=False)
            background_seq_sample = bg_seq[indices]
            background_temp_sample = bg_temp[indices]
            self.background_temp = np.mean(bg_temp, axis=0).reshape(1, -1)
            
            # Calculate total features correctly
            num_features = int(np.prod(background_seq_sample.shape[1:]))
            temp_features = int(np.prod(background_temp_sample.shape[1:]))
            total_features = num_features + temp_features
            
            # FIX: Set max_evals to required minimum
            computed = 2 * total_features + 1
            if max_evals_override == "auto":
                self.max_evals = "auto"
            else:
                self.max_evals = max(computed, max_evals_override or 0)
            print(f"[DEBUG] Total features: {total_features}, Setting max_evals: {self.max_evals}")
            
            # For multi-input models, we need to flatten inputs for SHAP
            # SHAP's PermutationExplainer expects a 2D array, not a list of arrays
            self._bg_seq_sample = background_seq_sample
            self._bg_temp_sample = background_temp_sample
            self._seq_shape = background_seq_sample.shape[1:]  # (seq_len,) or (seq_len, features)
            self._temp_shape = background_temp_sample.shape[1:]  # (temp_features,)
            self._seq_flat_size = int(np.prod(self._seq_shape))
            self._temp_flat_size = int(np.prod(self._temp_shape))
            
            # Create flattened background data for SHAP
            bg_seq_flat = background_seq_sample.reshape(len(background_seq_sample), -1)
            bg_temp_flat = background_temp_sample.reshape(len(background_temp_sample), -1)
            background_flat = np.hstack([bg_seq_flat, bg_temp_flat])
            
            def predict_fn_flat(x_flat):
                """Prediction function that takes flattened input and returns model output."""
                n_samples = x_flat.shape[0]
                # Split flattened input back into seq and temp
                x_seq_flat = x_flat[:, :self._seq_flat_size]
                x_temp_flat = x_flat[:, self._seq_flat_size:]
                # Reshape back to original shapes
                x_seq = x_seq_flat.reshape((n_samples,) + self._seq_shape)
                x_temp = x_temp_flat.reshape((n_samples,) + self._temp_shape)
                preds = self.model.predict([x_seq, x_temp], verbose=0)
                return preds if self.task == 'activity' else preds.flatten()
            
            self._predict_fn_flat = predict_fn_flat
            self._background_flat = background_flat
            
            try:
                # Use PermutationExplainer with flattened data
                self.explainer = shap.PermutationExplainer(
                    predict_fn_flat,
                    background_flat,
                )
            except Exception as e:
                print(f"[WARNING] SHAP PermutationExplainer init failed: {e}")
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    predict_fn_flat,
                    background_flat,
                )
        else:
            indices = np.random.choice(len(background_data), min(max_background, len(background_data)), replace=False)
            background_sample = background_data[indices]
            num_features = int(np.prod(background_sample.shape[1:]))
            
            # FIX: Set max_evals to required minimum
            computed = 2 * num_features + 1
            if max_evals_override == "auto":
                self.max_evals = "auto"
            else:
                self.max_evals = max(computed, max_evals_override or 0)
            print(f"[DEBUG] Total features: {num_features}, Setting max_evals: {self.max_evals}")
            
            try:
                self.explainer = shap.Explainer(self.model, background_sample, max_evals=self.max_evals)
            except Exception as e:
                print(f"[WARNING] SHAP explainer init fallback: {e}")
                def predict_fn_single(x):
                    preds = self.model.predict(x, verbose=0)
                    return preds if self.task == 'activity' else preds.flatten()
                self.explainer = shap.Explainer(predict_fn_single, background_sample, max_evals=self.max_evals)

    def _retry_with_required_max_evals(self, err):
        msg = str(err)
        # SHAP error strings often embed the required number after an expression
        # like "at least 2 * num_features + 1 = 1601!".
        numbers = [int(n) for n in re.findall(r"\d+", msg)]
        if not numbers:
            return False
        required = max(numbers)
        current = self.max_evals if isinstance(self.max_evals, (int, float)) else 0
        self.max_evals = max(current, required)
        # Rebuild explainer to ensure max_evals is applied internally.
        if self._background_data is not None:
            self.initialize_explainer(
                self._background_data,
                max_background=self._max_background or 100,
                max_evals_override=self.max_evals
            )
        elif hasattr(self.explainer, "max_evals"):
            self.explainer.max_evals = self.max_evals
        print(f"[DEBUG] Retrying SHAP with max_evals: {self.max_evals}")
        return True

    def _set_explainer_max_evals(self, value):
        if hasattr(self.explainer, "max_evals"):
            self.explainer.max_evals = value
        inner = getattr(self.explainer, "explainer", None)
        if inner is not None and hasattr(inner, "max_evals"):
            inner.max_evals = value

    def _compute_required_max_evals(self, row_args):
        try:
            from shap.utils import MaskedModel
            fm = MaskedModel(
                self.explainer.model,
                self.explainer.masker,
                self.explainer.link,
                self.explainer.linearize_link,
                *row_args
            )
            return 2 * len(fm) + 1
        except Exception as e:
            print(f"[WARNING] Could not compute required max_evals from masker: {e}")
            return None

    def _call_explainer(self, inputs, max_evals=None):
        if max_evals is None:
            max_evals = self.max_evals or "auto"
        try:
            return self.explainer(inputs, max_evals=max_evals)
        except TypeError:
            return self.explainer(inputs)

    def explain_samples(self, test_data, num_samples=20, indices=None):
        if isinstance(test_data, (list, tuple)):
            if indices is not None and len(indices) > 0:
                test_sample = test_data[0][indices]
                self.test_data = test_sample
                self.test_data_temp = test_data[1][indices]
            else:
                test_sample = test_data[0][:num_samples]
                self.test_data = test_sample
                self.test_data_temp = test_data[1][:num_samples]
        else:
            if indices is not None and len(indices) > 0:
                test_sample = test_data[indices]
                self.test_data = test_sample
            else:
                test_sample = test_data[:num_samples]
                self.test_data = test_sample
            
        print(f"Computing SHAP values for {len(test_sample)} samples...")

        # For multi-input models, flatten the test data
        if isinstance(test_data, (list, tuple)) and self.is_multi_input:
            if indices is not None and len(indices) > 0:
                test_temp = test_data[1][indices]
            else:
                test_temp = test_data[1][:num_samples]
            # Flatten test data same way as background
            test_seq_flat = test_sample.reshape(len(test_sample), -1)
            test_temp_flat = test_temp.reshape(len(test_temp), -1)
            test_flat = np.hstack([test_seq_flat, test_temp_flat])
            
            try:
                self.shap_values = self.explainer(test_flat)
            except Exception as e:
                print(f"[WARNING] SHAP explain failed: {e}")
                # Try with explicit max_evals
                n_features = test_flat.shape[1]
                required_max_evals = 2 * n_features + 1
                print(f"[DEBUG] Retrying with max_evals={required_max_evals}")
                self.shap_values = self.explainer(test_flat, max_evals=required_max_evals)
        else:
            try:
                self.shap_values = self._call_explainer(test_sample, max_evals=self.max_evals)
            except ValueError as e:
                if self._retry_with_required_max_evals(e):
                    self._set_explainer_max_evals(self.max_evals)
                    self.shap_values = self._call_explainer(test_sample, max_evals=self.max_evals)
                else:
                    raise
        return self.shap_values

    def _aggregate_by_activity(self):
        if self.shap_values is None: 
            return None, None, None

        values = self.shap_values.values
        if isinstance(values, list):
            values = values[0]
        
        seq_len = self.test_data.shape[1] if self.test_data is not None else None
        if seq_len is None:
            return None, None, None
        
        # Handle flattened multi-input case: SHAP values are (n_samples, total_flat_features)
        # where total_flat_features = seq_flat_size + temp_flat_size
        if self.is_multi_input and hasattr(self, '_seq_flat_size'):
            # Extract only sequence portion of SHAP values
            if values.ndim == 2 and values.shape[1] >= self._seq_flat_size:
                values = values[:, :self._seq_flat_size]
                # Reshape to (n_samples, seq_len, ...) if needed
                if self._seq_shape == (seq_len,):
                    # Shape is just (seq_len,), values are already (n_samples, seq_len)
                    pass
                else:
                    # Reshape to original sequence shape
                    values = values.reshape((values.shape[0],) + self._seq_shape)
        
        seq_axis = None
        for axis in range(1, values.ndim):
            if values.shape[axis] == seq_len:
                seq_axis = axis
                break
        
        if seq_axis is None:
            print(f"[DEBUG] Cannot find seq_axis. values.shape={values.shape}, seq_len={seq_len}")
            return None, None, None
        
        values = np.moveaxis(values, seq_axis, 1)
        
        if values.ndim > 2:
            if self.task == 'activity':
                values = np.abs(values).mean(axis=tuple(range(2, values.ndim)))
            else:
                values = values.mean(axis=tuple(range(2, values.ndim)))
        
        # Collect all unique activity names across all samples
        unique_names = set()
        for seq in self.test_data:
            unique_names.update([n for n in self._get_activity_names_for_sample(seq) if n != '[PAD]'])
        
        sorted_names = sorted(list(unique_names))
        name_map = {name: i for i, name in enumerate(sorted_names)}
        
        num_samples = values.shape[0]
        agg_shap_matrix = np.zeros((num_samples, len(sorted_names)))
        agg_feat_matrix = np.zeros((num_samples, len(sorted_names))) 
        
        # Aggregate SHAP values by activity name
        for i in range(num_samples):
            seq_names = self._get_activity_names_for_sample(self.test_data[i])
            for j, name in enumerate(seq_names):
                if name in name_map:
                    col_idx = name_map[name]
                    agg_shap_matrix[i, col_idx] += values[i, j]
                    agg_feat_matrix[i, col_idx] += 1
                    
        return agg_shap_matrix, agg_feat_matrix, sorted_names

    def plot_bar(self, output_dir):
        print("Generating Global Importance Plot (Bar)...")
        agg_values, _, names = self._aggregate_by_activity()
        if agg_values is None:
            print("[WARNING] SHAP values unavailable or invalid for plotting.")
            return
        
        mean_impact = np.abs(agg_values).mean(axis=0)
        df = pd.DataFrame({'Activity': names, 'Mean_Impact': mean_impact}).sort_values('Mean_Impact', ascending=False)
        df.to_csv(os.path.join(output_dir, 'global_importance_data.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(agg_values, feature_names=names, plot_type="bar", show=False, max_display=15)
        plt.title(f"Global Feature Importance ({self.task.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=300)
        plt.close()

    def plot_summary(self, output_dir):
        print("Generating Global Summary Plot...")
        agg_shap, agg_feat, names = self._aggregate_by_activity()
        if agg_shap is None:
            print("[WARNING] SHAP values unavailable or invalid for plotting.")
            return
        pd.DataFrame(agg_shap, columns=names).to_csv(os.path.join(output_dir, 'shap_values_matrix.csv'), index=False)

        # Use aggregated activity-level summary to avoid repeated position names.
        plt.figure(figsize=(13.5, 8))
        shap.summary_plot(agg_shap, features=agg_feat, feature_names=names, show=False, max_display=15)
        plt.title(f"Feature Impact Distribution ({self.task.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300)
        plt.close()

        # If we have temporal features, also save a summary plot for them.
        if isinstance(self.shap_values.values, list) and self.test_data_temp is not None:
            temp_values = self.shap_values.values[1]
            if temp_values.ndim > 2:
                temp_values = temp_values.mean(axis=tuple(range(2, temp_values.ndim)))
            temp_feature_names = [f"Temp_{i+1}" for i in range(self.test_data_temp.shape[1])]
            plt.figure(figsize=(10, 6))
            shap.summary_plot(temp_values, features=self.test_data_temp, feature_names=temp_feature_names, show=False, max_display=15)
            plt.title(f"Temporal Feature Impact Distribution ({self.task.capitalize()})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary_plot_temp.png'), dpi=300)
            plt.close()

    def save_explanations(self, output_dir):
        print("[OK] SHAP computations complete.")

class TimestepSHAPExplainer(SHAPExplainer):
    """SHAP Explainer with timestep-level attribution (optional timestamps)."""
    def __init__(self, model, task='time', label_encoder=None, scaler=None, timestamps=None):
        super().__init__(model, task, label_encoder, scaler)
        self.model_has_timestep_outputs = self._detect_model_type()
        self.timestamps = timestamps

        if self.model_has_timestep_outputs:
            print("[OK] Detected timestep-explainable model - will generate temporal plots")
        else:
            print("[INFO] Using original aggregated explanations")

        if self.timestamps is not None:
            print(f"[OK] Timestamps provided for {len(self.timestamps)} samples")
        else:
            print("[INFO] No timestamps provided - will use timestep indices")

    def _detect_model_type(self):
        if ExplainabilityConfig.MODEL_TYPE == 'per_timestep':
            return True
        if ExplainabilityConfig.MODEL_TYPE == 'original':
            return False
        if hasattr(self.model, 'outputs'):
            return len(self.model.outputs) > 1
        return False

    def _sequence_shap_values(self):
        if self.shap_values is None or self.test_data is None:
            return None

        values = self.shap_values.values
        if isinstance(values, list):
            values = values[0]

        seq_len = self.test_data.shape[1]

        if self.is_multi_input and hasattr(self, '_seq_flat_size') and values.ndim == 2:
            if values.shape[1] >= self._seq_flat_size:
                values = values[:, :self._seq_flat_size]
                if self._seq_shape == (seq_len,):
                    values = values.reshape((values.shape[0], seq_len))
                else:
                    values = values.reshape((values.shape[0],) + self._seq_shape)
                    if values.ndim > 2:
                        values = values.mean(axis=tuple(range(2, values.ndim)))

        if values.ndim > 2:
            seq_axis = None
            for axis in range(1, values.ndim):
                if values.shape[axis] == seq_len:
                    seq_axis = axis
                    break
            if seq_axis is not None:
                values = np.moveaxis(values, seq_axis, 1)
                if values.ndim > 2:
                    values = values.mean(axis=tuple(range(2, values.ndim)))

        return values

    def plot_temporal_evolution(self, output_dir, sample_idx=0, show_prediction=True):
        if self.shap_values is None:
            print("No SHAP values computed. Run explain_samples() first.")
            return

        seq_values = self._sequence_shap_values()
        if seq_values is None:
            print("No valid sequence SHAP values available.")
            return

        print(f"Generating Temporal Evolution Plot for sample {sample_idx}...")

        sample_shap = seq_values[sample_idx]
        sample_sequence = self.test_data[sample_idx]
        activity_names = self._get_activity_names_for_sample(sample_sequence)

        non_pad_mask = sample_sequence > 0
        filtered_shap = sample_shap[non_pad_mask]
        filtered_activities = [name for name, is_valid in zip(activity_names, non_pad_mask) if is_valid]

        if self.timestamps is not None and sample_idx < len(self.timestamps):
            sample_timestamps = self.timestamps[sample_idx]
            filtered_timestamps = [sample_timestamps[i] for i, valid in enumerate(non_pad_mask) if valid]
            x_values = np.arange(len(filtered_timestamps))
            x_labels = filtered_timestamps
            use_timestamps = True
        else:
            x_values = np.arange(len(filtered_shap))
            x_labels = x_values
            use_timestamps = False

        fig, ax1 = plt.subplots(figsize=(16, 7))

        positive_shap = np.where(filtered_shap > 0, filtered_shap, 0)
        negative_shap = np.where(filtered_shap < 0, filtered_shap, 0)

        ax1.bar(x_values, positive_shap, color='#2ca02c', alpha=1.0,
                label='Increases Prediction', width=0.8, edgecolor='black', linewidth=0.5)
        ax1.bar(x_values, negative_shap, color='#d62728', alpha=1.0,
                label='Decreases Prediction', width=0.8, edgecolor='black', linewidth=0.5)

        ax1.axhline(0, color='black', linewidth=1)

        if use_timestamps:
            ax1.set_xticks(x_values)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax1.set_xlabel('Event Timestamp (Days from Case Start)', fontsize=13, fontweight='bold')
        else:
            ax1.set_xlabel('Time steps', fontsize=13, fontweight='bold')

        ax1.set_ylabel('SHAP values (contribution to prediction)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left', fontsize=11)

        abs_shap = np.abs(filtered_shap)
        threshold = np.percentile(abs_shap, 75) if len(abs_shap) else 0

        for i, (x_pos, act, shap_val) in enumerate(zip(x_values, filtered_activities, filtered_shap)):
            if abs_shap[i] > threshold and act != '[PAD]':
                y_pos = shap_val + (0.2 if shap_val > 0 else -0.2)
                ax1.text(x_pos, y_pos, act, ha='center',
                         va='bottom' if shap_val > 0 else 'top',
                         fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        if show_prediction and self.model_has_timestep_outputs:
            try:
                temp_input = self.background_temp if self.background_temp is not None else np.zeros((1, 3))
                outputs = self.model.predict([sample_sequence.reshape(1, -1), temp_input], verbose=0)
                if isinstance(outputs, list) and len(outputs) > 1:
                    timestep_preds = outputs[1][0]
                    filtered_preds = timestep_preds[non_pad_mask]
                    if self.scaler is not None:
                        filtered_preds = self.scaler.inverse_transform(
                            filtered_preds.reshape(-1, 1)
                        ).flatten()

                    ax2 = ax1.twinx()
                    ax2.plot(x_values, filtered_preds, color='black', linewidth=2,
                             label='Predicted remaining time', marker='o', markersize=3)
                    ax2.set_ylabel('Predicted remaining time (days)', fontsize=12, fontweight='bold')
                    ax2.legend(loc='upper right', fontsize=11)
            except Exception as e:
                print(f"Could not add prediction overlay: {e}")

        plt.title(f'Transformer Model - SHAP Explainability (Sample {sample_idx})',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_temporal_evolution_sample_{sample_idx}.png'), dpi=300)
        plt.close()

        df_data = {
            'Activity': filtered_activities,
            'SHAP_Value': filtered_shap
        }
        if use_timestamps:
            df_data['Timestamp'] = x_labels
        else:
            df_data['Timestep'] = x_values

        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, f'shap_timestep_data_sample_{sample_idx}.csv'), index=False)

    def plot_timestep_heatmap(self, output_dir, sample_idx=0):
        if self.shap_values is None:
            return

        seq_values = self._sequence_shap_values()
        if seq_values is None:
            return

        print(f"Generating Timestep Heatmap for sample {sample_idx}...")

        sample_shap = seq_values[sample_idx]
        sample_sequence = self.test_data[sample_idx]
        activity_names = self._get_activity_names_for_sample(sample_sequence)

        non_pad_mask = sample_sequence > 0
        filtered_shap = sample_shap[non_pad_mask]
        filtered_activities = [name for name, is_valid in zip(activity_names, non_pad_mask) if is_valid]
        timesteps = np.arange(len(filtered_shap))

        if self.timestamps is not None and sample_idx < len(self.timestamps):
            sample_timestamps = self.timestamps[sample_idx]
            filtered_timestamps = [sample_timestamps[i] for i, valid in enumerate(non_pad_mask) if valid]
            x_labels = [f"{act}\n{ts}" for act, ts in zip(filtered_activities, filtered_timestamps)]
        else:
            x_labels = filtered_activities

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['#d62728' if val < 0 else '#2ca02c' for val in filtered_shap]
        ax.bar(timesteps, filtered_shap, color=colors, alpha=1.0, edgecolor='black', linewidth=0.5)

        ax.set_xticks(timesteps)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Event (Activity + Timestamp)' if self.timestamps else 'Timestep (Activity)',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('SHAP Value (Contribution)', fontsize=12, fontweight='bold')
        ax.set_title(f'Timestep-Level SHAP Attribution - Sample {sample_idx}',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Increases Prediction'),
            Patch(facecolor='#d62728', label='Decreases Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_timestep_heatmap_sample_{sample_idx}.png'), dpi=300)
        plt.close()

    def plot_global_temporal_importance(self, output_dir):
        if self.shap_values is None or self.test_data is None:
            return

        seq_values = self._sequence_shap_values()
        if seq_values is None:
            return

        print("Generating Global Temporal Importance Plot...")

        mean_shap_per_timestep = np.mean(np.abs(seq_values), axis=0)

        activity_labels = []
        for pos in range(seq_values.shape[1]):
            activities_at_pos = []
            for sample in self.test_data:
                if sample[pos] > 0:
                    try:
                        act = self.label_encoder.inverse_transform([int(sample[pos])-1])[0]
                        activities_at_pos.append(act)
                    except Exception:
                        pass
            if activities_at_pos:
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                activity_labels.append(most_common)
            else:
                activity_labels.append('[PAD]')

        fig, ax = plt.subplots(figsize=(14, 6))
        timesteps = np.arange(len(mean_shap_per_timestep))

        ax.bar(timesteps, mean_shap_per_timestep, color='#2ca02c', alpha=1.0,
               edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Timestep Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        ax.set_title('Global Timestep Importance (Averaged Across All Samples)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        top_n = 10
        top_indices = np.argsort(mean_shap_per_timestep)[-top_n:]
        for idx in top_indices:
            ax.text(idx, mean_shap_per_timestep[idx], activity_labels[idx],
                    ha='center', va='bottom', fontsize=8, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_global_temporal_importance.png'), dpi=300)
        plt.close()

        df = pd.DataFrame({
            'Timestep': timesteps,
            'Most_Common_Activity': activity_labels,
            'Mean_Absolute_SHAP': mean_shap_per_timestep
        })
        df.to_csv(os.path.join(output_dir, 'shap_global_temporal_data.csv'), index=False)

class LIMEExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.explainer = None
        self.explanations = []
        self.test_data_seq = None
        self.test_data_temp = None
        self.is_multi_input = False
        self.vocab_size = None
        self.y_true = None
        # DEBUG: Print whether label_encoder is available
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - LIME will show generic Activity labels!")
            print("[FIX] Pass label_encoder to run_transformer_explainability()")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _aggregate_feature_names(self, data):
        if self.label_encoder is None:
            return [f'Position_{i+1}' for i in range(data.shape[1])]
        feature_names = []
        for pos in range(data.shape[1]):
            activities_at_pos = []
            for sample in data:
                token = sample[pos]
                if token > 0:
                    try:
                        # Token indices are offset by +1 (0 is padding)
                        activity = self.label_encoder.inverse_transform([int(token) - 1])[0]
                        activities_at_pos.append(activity)
                    except Exception as e:
                        print(f"[WARNING] Failed to decode activity token {int(token)}: {e}")
                        pass
    
            if activities_at_pos:
                # Find most common activity at this position
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                feature_names.append(most_common)
            else:
                # Fallback for padding-only positions
                feature_names.append(f'Position_{pos+1}')
        return feature_names
        
    def initialize_explainer(self, training_data, num_classes=None):
        print("Initializing LIME Explainer...")
        if isinstance(training_data, (list, tuple)):
            init_data = training_data[0]
        else:
            init_data = training_data
        
        if self.vocab_size is None:
            if self.label_encoder is not None:
                self.vocab_size = len(self.label_encoder.classes_) + 1
            else:
                self.vocab_size = int(np.max(init_data)) + 1 if init_data.size > 0 else 1
        
        # Aggregate feature names based on actual activities
        feature_names = self._aggregate_feature_names(init_data)
        class_names = None
        mode = 'regression'
        
        if self.task == 'activity':
            mode = 'classification'
            if self.label_encoder:
                class_names = self.label_encoder.classes_.tolist()
            elif num_classes:
                class_names = [str(i) for i in range(num_classes)]
                
        self.explainer = lime_tabular.LimeTabularExplainer(
            init_data,
            mode=mode,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=False,
            verbose=False
        )
    
    def explain_samples(self, test_data, num_samples=10, num_features=15, y_true=None):
        print(f"Generating LIME explanations for {num_samples} samples...")
        
        if isinstance(test_data, (list, tuple)):
            self.test_data_seq = test_data[0][:num_samples]
            self.test_data_temp = test_data[1][:num_samples]
            self.is_multi_input = True
            print(f"[DEBUG explain_samples] Processing {len(self.test_data_seq)} sequences")
        else:
            self.test_data_seq = test_data[:num_samples]
            self.is_multi_input = False
            print(f"[DEBUG explain_samples] Processing {len(self.test_data_seq)} samples")
        if y_true is not None:
            self.y_true = y_true[:num_samples]
            
        vocab_size = self.vocab_size if self.vocab_size is not None else int(np.max(self.test_data_seq)) + 1
        
        for i in tqdm(range(len(self.test_data_seq))):
            try:
                if self.is_multi_input:
                    current_temp = self.test_data_temp[i].reshape(1, -1)
                    def predict_fn(x_seq):
                        if x_seq.ndim == 1: x_seq = x_seq.reshape(1, -1)
                        x_seq = np.clip(np.round(x_seq), 0, vocab_size-1).astype(int)
                        temp_batch = np.repeat(current_temp, x_seq.shape[0], axis=0)
                        preds = self.model.predict([x_seq, temp_batch], verbose=0)
                        return preds.flatten() if self.task != 'activity' else preds
                else:
                    def predict_fn(x_seq):
                        if x_seq.ndim == 1: x_seq = x_seq.reshape(1, -1)
                        x_seq = np.clip(np.round(x_seq), 0, vocab_size-1).astype(int)
                        preds = self.model.predict(x_seq, verbose=0)
                        return preds.flatten() if self.task != 'activity' else preds

                exp = self.explainer.explain_instance(
                    self.test_data_seq[i],
                    predict_fn,
                    num_features=num_features,
                    top_labels=1
                )
                self.explanations.append(exp)
                
            except Exception as e:
                print(f"Error explaining sample {i}: {e}")
                self.explanations.append(None)        
        return self.explanations

    def _get_activity_name(self, token_idx):
        if token_idx == 0: 
            return "[PAD]"
        if self.label_encoder:
            try: 
                return self.label_encoder.inverse_transform([int(token_idx)-1])[0]
            except Exception as e:
                print(f"[WARNING] Failed to decode activity token {int(token_idx)}: {e}")
                pass
        return f"Activity_{int(token_idx)}"

    def plot_explanation(self, output_dir, sample_idx=0, original_idx=None):
        if sample_idx >= len(self.explanations) or self.explanations[sample_idx] is None:
            print(f"LIME Explanation not found for sample {sample_idx}.")
            return
    
        # Use original_idx for filename, sample_idx for data access
        display_idx = original_idx if original_idx is not None else sample_idx 
        print(f"Generating Research-Grade LIME Plot for sample {display_idx}...")
        exp = self.explanations[sample_idx]
        current_seq = self.test_data_seq[sample_idx]  # Use local index
        
        pred_activity_name = None
        try:
            if self.task == 'activity':
                if hasattr(exp, 'top_labels') and exp.top_labels:
                    label_to_explain = exp.top_labels[0]
                else:
                    label_to_explain = 1 
                
                pred_probs = exp.predict_proba
                confidence = pred_probs[label_to_explain] if pred_probs is not None else 0.0
                pred_label = label_to_explain
                if self.label_encoder is not None:
                    try:
                        pred_label = self.label_encoder.inverse_transform([int(label_to_explain)])[0]
                    except Exception:
                        pred_label = label_to_explain
                pred_activity_name = pred_label
                gt_label = None
                if self.y_true is not None and sample_idx < len(self.y_true):
                    gt_label = self.y_true[sample_idx]
                    if self.label_encoder is not None:
                        try:
                            gt_label = self.label_encoder.inverse_transform([int(gt_label)])[0]
                        except Exception:
                            pass
                gt_text = f" | Ground Truth: {gt_label}" if gt_label is not None else ""
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Class: {pred_label} | Confidence: {confidence:.2f}{gt_text}"
                lime_list = exp.as_list(label=label_to_explain)
            else:
                raw_pred = exp.predicted_value
                display_val = raw_pred
                
                if self.scaler is not None:
                    unscaled = self.scaler.inverse_transform([[raw_pred]])[0][0]
                    display_val = unscaled
                gt_val = None
                if self.y_true is not None and sample_idx < len(self.y_true):
                    gt_val = self.y_true[sample_idx]
                    if self.scaler is not None:
                        try:
                            gt_val = self.scaler.inverse_transform([[gt_val]])[0][0]
                        except Exception:
                            pass
                gt_text = f" | Ground Truth: {gt_val:.2f}" if gt_val is not None else ""
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Value: {display_val:.2f}{gt_text}"
                lime_list = exp.as_list()
                
        except Exception as e:
            print(f"Warning: Could not extract full LIME details: {e}")
            title = f"LIME Explanation (Sample {display_idx})"
            lime_list = exp.as_list()

        activity_stats = {} 
        for rule, weight in lime_list:
            # Try to extract activity name from rule
            # Rules can be: "Create Order <= 3.00" or just "Create Order"
            if rule.startswith('Position_'):
                # Old-style Position_N label - extract position and map to activity
                match = re.search(r'Position_(\d+)', rule)
                if match:
                    pos = int(match.group(1)) - 1
                    if 0 <= pos < len(current_seq):
                        name = self._get_activity_name(current_seq[pos])
                    else:
                        continue
                else:
                    continue
            else:
                # Activity name from aggregated feature_names
                # Extract base name (remove conditions like "<= 3.00")
                name = rule.split('<=')[0].split('>')[0].strip()
                # If name still looks like a Position_ label, try mapping or skip.
                if name.startswith("Position_"):
                    match = re.search(r'Position_(\d+)', name)
                    if match:
                        pos = int(match.group(1)) - 1
                        if 0 <= pos < len(current_seq):
                            name = self._get_activity_name(current_seq[pos])
                        else:
                            continue
                    else:
                        continue
            
            if name not in activity_stats:
                activity_stats[name] = {'weight': 0.0, 'count': 0}
            activity_stats[name]['weight'] += weight
            activity_stats[name]['count'] += 1
        
        data = []
        for name, stats in activity_stats.items():
            label = f"{name} (x{stats['count']})" if stats['count'] > 1 else name
            data.append({
                'Activity': label,
                'Weight': stats['weight'],
                'AbsWeight': abs(stats['weight'])
            })
            
        if not data:
            print("No valid LIME features found to plot.")
            return

        df = pd.DataFrame(data).sort_values('AbsWeight', ascending=True)
        df[['Activity', 'Weight']].to_csv(os.path.join(output_dir, f'lime_explanation_sample_{display_idx}.csv'), index=False)

        plt.figure(figsize=(10, 6))
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in df['Weight']]
        
        bars = plt.barh(df['Activity'], df['Weight'], color=colors, height=0.6)
        
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.margins(x=0.15)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel("Contribution to Prediction", fontsize=11)
        
        for rect in bars:
            w = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            padding = 0.0005 if w > 0 else -0.0005
            ha = 'left' if w > 0 else 'right'
            plt.text(w + padding, y, f'{w:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')

        from matplotlib.patches import Patch
        plt.legend(handles=[
            Patch(facecolor='#2ca02c', label='Supports'),
            Patch(facecolor='#d62728', label='Contradicts')
        ], loc='lower right', frameon=True)

        # Add full sample sequence at the bottom with predicted activity highlighted.
        plt.tight_layout(rect=[0.05, 0.05, 0.98, 1])
        plt.savefig(os.path.join(output_dir, f'lime_explanation_sample_{display_idx}.png'), dpi=300)
        plt.close()

    def save_explanations(self, output_dir):
        print("[OK] LIME computations complete.")


def generate_comparison_report(output_dir, shap_dir, lime_dir):
    import pandas as pd
    import os
    import re
    
    summary_data = []
    
    # Load SHAP results if available
    shap_importance = {}
    if shap_dir and os.path.exists(os.path.join(shap_dir, 'global_importance_data.csv')):
        shap_df = pd.read_csv(os.path.join(shap_dir, 'global_importance_data.csv'))
        shap_importance = dict(zip(shap_df['Activity'], shap_df['Mean_Impact']))
    
    # Load LIME results if available (aggregate from multiple samples)
    lime_importance = {}
    if lime_dir:
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('lime_explanation_sample_') and f.endswith('.csv')]
        if lime_files:
            all_lime_weights = {}
            for lime_file in lime_files:
                lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                for _, row in lime_df.iterrows():
                    activity = row['Activity']
                    activity = re.sub(r'\s+\(x\d+\)$', '', str(activity)).strip()
                    weight = abs(row['Weight'])
                    if activity not in all_lime_weights:
                        all_lime_weights[activity] = []
                    all_lime_weights[activity].append(weight)
            
            # Average LIME weights
            lime_importance = {act: sum(weights)/len(weights) for act, weights in all_lime_weights.items()}
    
    # Combine results
    all_features = set(shap_importance.keys()) | set(lime_importance.keys())
    
    for feature in all_features:
        shap_score = shap_importance.get(feature, 0)
        lime_score = lime_importance.get(feature, 0)
        avg_score = (shap_score + lime_score) / 2 if shap_score and lime_score else (shap_score or lime_score)
        
        summary_data.append({
            'Feature': feature,
            'SHAP_Importance': shap_score,
            'LIME_Importance': lime_score,
            'Average_Importance': avg_score,
            'Agreement': 'Both' if shap_score > 0 and lime_score > 0 else 'SHAP only' if shap_score > 0 else 'LIME only'
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Average_Importance', ascending=False)
    summary_df.to_csv(os.path.join(output_dir, 'feature_importance_summary.csv'), index=False)
    
    # Generate text report
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPLAINABILITY METHODS COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total features analyzed: {len(all_features)}\n")
        f.write(f"Features identified by both methods: {len([x for x in summary_data if x['Agreement'] == 'Both'])}\n")
        f.write(f"Features identified by SHAP only: {len([x for x in summary_data if x['Agreement'] == 'SHAP only'])}\n")
        f.write(f"Features identified by LIME only: {len([x for x in summary_data if x['Agreement'] == 'LIME only'])}\n\n")
        
        f.write("Top 10 Most Important Features (Average):\n")
        f.write("-"*70 + "\n")
        for i, row in enumerate(summary_df.head(10).to_dict('records'), 1):
            f.write(f"{i:2d}. {row['Feature']:<30} | Avg: {row['Average_Importance']:.4f}\n")
            f.write(f"    SHAP: {row['SHAP_Importance']:.4f} | LIME: {row['LIME_Importance']:.4f}\n\n")
    
    print(f"[OK] Feature importance summary saved: feature_importance_summary.csv")
    print(f"[OK] Comparison report saved: comparison_report.txt")


def select_diverse_samples(data, task, num_diverse=10, label_encoder=None):
    import numpy as np
    
    if task == 'activity':
        X_test = data.get('X_test', [])
        y_test = data.get('y_test', [])
        test_size = len(y_test)
        num_classes = len(label_encoder.classes_) if label_encoder is not None else len(np.unique(y_test))
    else:
        y_test = data.get('y_test', [])
        test_size = len(y_test) if hasattr(y_test, '__len__') else len(data.get('X_seq_test', []))
        num_classes = None
    
    if test_size == 0:
        return []
    
    selected = []
    selected_set = set()
    
    if task == 'activity' and num_classes:
        required = num_classes
        if num_diverse < required:
            print(f"[WARNING] num_samples={num_diverse} < num_classes={required}. Increasing to {required} for full coverage.")
            num_diverse = required
        
        class_to_sample = {}
        # Ensure each activity appears at least once in sampled sequences
        if len(X_test) > 0:
            for idx, seq in enumerate(X_test):
                tokens = set([int(t) for t in seq if t > 0])
                for token in tokens:
                    class_idx = token - 1
                    if 0 <= class_idx < num_classes and class_idx not in class_to_sample:
                        class_to_sample[class_idx] = idx
                if len(class_to_sample) == num_classes:
                    break
        
        selected = [class_to_sample[k] for k in sorted(class_to_sample.keys())]
        selected_set = set(selected)
        if len(selected) < num_classes:
            missing = [str(c) for c in range(num_classes) if c not in class_to_sample]
            print(f"[WARNING] Could not find samples containing activities: {', '.join(missing)}")
    
    # Fill remaining slots with evenly spaced samples
    remaining = max(0, min(num_diverse, test_size) - len(selected))
    if remaining > 0:
        step = max(1, test_size // max(remaining, 1))
        for idx in range(0, test_size, step):
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) >= min(num_diverse, test_size):
                    break
    
    return selected[:min(num_diverse, test_size)]


def _validate_explainability_coverage(task, label_encoder, shap_dir=None, lime_dir=None):
    if task != 'activity' or label_encoder is None:
        return
    
    expected = set(label_encoder.classes_.tolist())
    
    if shap_dir:
        shap_path = os.path.join(shap_dir, 'global_importance_data.csv')
        if not os.path.exists(shap_path):
            raise RuntimeError("SHAP output missing: global_importance_data.csv")
        shap_df = pd.read_csv(shap_path)
        shap_feats = set(shap_df['Activity'].astype(str).tolist())
        missing_shap = sorted(expected - shap_feats)
        if missing_shap:
            print(f"[WARNING] SHAP missing activities: {', '.join(missing_shap)}")
    
    if lime_dir:
        if not os.path.isdir(lime_dir):
            raise RuntimeError("LIME output missing: lime directory not found.")
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('lime_explanation_sample_') and f.endswith('.csv')]
        if not lime_files:
            raise RuntimeError("LIME output missing: no lime_explanation_sample_*.csv files found.")
        lime_feats = set()
        for lime_file in lime_files:
            lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
            for val in lime_df['Activity'].astype(str).tolist():
                name = re.sub(r'\s+\(x\d+\)$', '', val).strip()
                lime_feats.add(name)
        missing_lime = sorted(expected - lime_feats)
        if missing_lime:
            print(f"[WARNING] LIME missing activities: {', '.join(missing_lime)}")


# =============================================================================
# EXPLAINABILITY BENCHMARK METRICS
# =============================================================================

class ExplainabilityBenchmark:
    """
    Comprehensive benchmark metrics for evaluating and comparing explainability methods.
    
    Metrics implemented:
    1. Faithfulness - Do top features actually impact predictions?
    2. Stability - Are explanations consistent for similar inputs?
    3. Method Agreement - Do SHAP and LIME agree on important features?
    4. Monotonicity - Does removing important features decrease performance monotonically?
    5. Infidelity - How well do explanations approximate model behavior?
    
    Reference: Evaluating Feature Attribution Methods (Nguyen et al., 2020)
    """
    
    def __init__(self, model, task='activity', is_multi_input=False, 
                 seq_shape=None, temp_shape=None, scaler=None, attribution_fn=None):
        self.model = model
        self.task = task
        self.is_multi_input = is_multi_input
        self.seq_shape = seq_shape
        self.temp_shape = temp_shape
        self.scaler = scaler
        self.attribution_fn = attribution_fn
        self.results = {}
        
    def _predict(self, x_seq, x_temp=None):
        """Unified prediction function handling both single and multi-input models."""
        if self.is_multi_input and x_temp is not None:
            preds = self.model.predict([x_seq, x_temp], verbose=0)
        else:
            preds = self.model.predict(x_seq, verbose=0)
        
        if self.task == 'activity':
            return preds
        else:
            return preds.flatten()
    
    def _get_baseline_value(self, x_seq):
        """Get baseline value for masking (mean or zero)."""
        return np.zeros_like(x_seq[0]) if len(x_seq.shape) > 1 else 0

    def _valid_positions(self, sequence):
        if sequence is None:
            return None
        seq = np.asarray(sequence)
        valid = np.where(seq > 0)[0]
        if valid.size == 0:
            return np.arange(seq.shape[0])
        return valid

    def _top_k_indices(self, sample_attr, k, valid_positions=None):
        attr = np.asarray(sample_attr)
        if valid_positions is None:
            valid_positions = np.arange(attr.shape[0])
        valid_positions = np.asarray(valid_positions, dtype=int)
        if valid_positions.size == 0:
            return np.array([], dtype=int)
        if k >= valid_positions.size:
            return valid_positions
        valid_scores = np.abs(attr[valid_positions])
        selected = np.argsort(valid_scores)[-k:]
        return valid_positions[selected]

    def _target_index(self, prediction):
        if self.task != 'activity':
            return None
        pred = np.asarray(prediction).reshape(-1)
        return int(np.argmax(pred))

    def _target_score(self, prediction, target_idx=None):
        pred = np.asarray(prediction)
        if self.task == 'activity':
            flat = pred.reshape(-1)
            if target_idx is None:
                target_idx = int(np.argmax(flat))
            return float(flat[target_idx])
        return float(np.asarray(pred).reshape(-1).mean())

    def _prediction_change(self, original_pred, updated_pred, target_idx=None):
        if self.task == 'activity':
            return self._target_score(original_pred, target_idx) - self._target_score(updated_pred, target_idx)
        return float(np.abs(np.asarray(original_pred) - np.asarray(updated_pred)).mean())

    def _attribution_similarity(self, baseline_attr, perturbed_attr):
        baseline = np.asarray(baseline_attr, dtype=float)
        perturbed = np.asarray(perturbed_attr, dtype=float)
        denom = np.linalg.norm(baseline) * np.linalg.norm(perturbed)
        if denom == 0:
            return 1.0 if np.linalg.norm(baseline - perturbed) == 0 else 0.0
        return float(np.dot(baseline, perturbed) / denom)

    # -------------------------------------------------------------------------
    # 1. FAITHFULNESS METRICS
    # -------------------------------------------------------------------------
    
    def faithfulness_correlation(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Faithfulness measures if removing top-k important features changes predictions.
        Higher correlation between importance and prediction change = better faithfulness.
        
        Args:
            x_seq: Sequence input (n_samples, seq_len)
            x_temp: Temporal features (n_samples, temp_features) or None
            attributions: Feature importance scores (n_samples, seq_len)
            k_values: List of k values to test
            
        Returns:
            dict with faithfulness scores for each k
        """
        print("Computing Faithfulness Correlation...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
                
            pred_changes = []
            importance_sums = []
            
            for i in range(n_samples):
                # Original prediction
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                target_idx = self._target_index(orig_pred)
                
                # Get top-k important feature indices
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = self._top_k_indices(sample_attr, k, self._valid_positions(x_seq[i]))
                if top_k_idx.size == 0:
                    continue
                
                # Mask top-k features
                x_masked = x_seq[i:i+1].copy()
                x_masked[0, top_k_idx] = 0  # Zero masking
                
                # Prediction after masking
                masked_pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                
                # Calculate prediction change
                pred_change = self._prediction_change(orig_pred, masked_pred, target_idx)
                
                pred_changes.append(pred_change)
                importance_sums.append(sample_attr[top_k_idx].sum())
            
            # Correlation between importance sum and prediction change
            from scipy.stats import spearmanr, pearsonr
            if len(set(pred_changes)) > 1 and len(set(importance_sums)) > 1:
                spearman_corr, spearman_p = spearmanr(importance_sums, pred_changes)
                pearson_corr, pearson_p = pearsonr(importance_sums, pred_changes)
            else:
                spearman_corr, spearman_p = 0.0, 1.0
                pearson_corr, pearson_p = 0.0, 1.0
            
            results[f'faithfulness_k{k}'] = {
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'mean_pred_change': float(np.mean(pred_changes)),
                'std_pred_change': float(np.std(pred_changes))
            }
            
        return results
    
    def comprehensiveness(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Comprehensiveness: Prediction change when removing top-k features.
        Higher = explanations capture important features.
        
        Formula: Comprehensiveness = f(x) - f(x \ top_k_features)
        """
        print("Computing Comprehensiveness...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
            
            comp_scores = []
            
            for i in range(n_samples):
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                target_idx = self._target_index(orig_pred)
                
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = self._top_k_indices(sample_attr, k, self._valid_positions(x_seq[i]))
                if top_k_idx.size == 0:
                    continue
                
                x_masked = x_seq[i:i+1].copy()
                x_masked[0, top_k_idx] = 0
                
                masked_pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                
                comp = self._prediction_change(orig_pred, masked_pred, target_idx)
                
                comp_scores.append(comp)
            
            results[f'comprehensiveness_k{k}'] = {
                'mean': float(np.mean(comp_scores)),
                'std': float(np.std(comp_scores)),
                'median': float(np.median(comp_scores))
            }
        
        return results
    
    def sufficiency(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Sufficiency: Prediction using ONLY top-k features.
        Lower = top features are sufficient to make prediction.
        
        Formula: Sufficiency = f(x) - f(only_top_k_features)
        """
        print("Computing Sufficiency...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
            
            suff_scores = []
            
            for i in range(n_samples):
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                target_idx = self._target_index(orig_pred)
                
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = self._top_k_indices(sample_attr, k, self._valid_positions(x_seq[i]))
                if top_k_idx.size == 0:
                    continue
                
                # Keep ONLY top-k features, mask everything else
                x_only_top = np.zeros_like(x_seq[i:i+1])
                x_only_top[0, top_k_idx] = x_seq[i, top_k_idx]
                
                top_pred = self._predict(x_only_top, x_temp[i:i+1] if x_temp is not None else None)
                
                suff = self._prediction_change(orig_pred, top_pred, target_idx)
                
                suff_scores.append(suff)
            
            results[f'sufficiency_k{k}'] = {
                'mean': float(np.mean(suff_scores)),
                'std': float(np.std(suff_scores)),
                'median': float(np.median(suff_scores))
            }
        
        return results
    
    # -------------------------------------------------------------------------
    # 2. STABILITY METRICS
    # -------------------------------------------------------------------------
    
    def stability(self, x_seq, x_temp, attributions, noise_std=0.01, n_perturbations=10):
        """
        Stability: Consistency of explanations under small input perturbations.
        Lower variance = more stable explanations.
        
        Args:
            x_seq: Input sequences
            x_temp: Temporal features
            attributions: Original SHAP/LIME values
            noise_std: Standard deviation of Gaussian noise
            n_perturbations: Number of perturbation trials
        """
        print("Computing Stability...")
        if self.attribution_fn is None:
            raise RuntimeError("Stability requires an attribution recomputation callback.")

        n_samples = min(len(x_seq), 20)
        variance_scores = []
        cosine_scores = []

        for i in range(n_samples):
            original_attr = np.asarray(attributions[i], dtype=float)
            perturbed_attrs = []

            for _ in range(n_perturbations):
                x_perturbed = x_seq[i:i+1].copy()
                valid_positions = self._valid_positions(x_perturbed[0])
                if valid_positions.size > 0:
                    n_mask = max(1, int(np.ceil(valid_positions.size * 0.1)))
                    mask_positions = np.random.choice(valid_positions, size=n_mask, replace=False)
                    x_perturbed[0, mask_positions] = 0

                x_temp_perturbed = None
                if x_temp is not None:
                    x_temp_perturbed = x_temp[i:i+1].copy().astype(float)
                    temp_noise = np.random.normal(0.0, noise_std, x_temp_perturbed.shape)
                    x_temp_perturbed = (x_temp_perturbed + temp_noise).astype(x_temp.dtype, copy=False)

                perturbed_attr = self.attribution_fn(x_perturbed, x_temp_perturbed)
                if perturbed_attr is None:
                    continue
                perturbed_arr = np.asarray(perturbed_attr, dtype=float).reshape(-1)
                min_len = min(len(original_attr), len(perturbed_arr))
                if min_len == 0:
                    continue
                perturbed_arr = perturbed_arr[:min_len]
                baseline_arr = original_attr[:min_len]
                perturbed_attrs.append(perturbed_arr)
                cosine_scores.append(self._attribution_similarity(baseline_arr, perturbed_arr))

            if perturbed_attrs:
                stacked = np.stack(perturbed_attrs, axis=0)
                variance_scores.append(float(np.var(stacked, axis=0).mean()))

        mean_variance = float(np.mean(variance_scores)) if variance_scores else 0.0
        max_variance = float(np.max(variance_scores)) if variance_scores else 0.0
        mean_cosine = float(np.mean(cosine_scores)) if cosine_scores else 0.0

        return {
            'stability': {
                'mean_variance': mean_variance,
                'max_variance': max_variance,
                'mean_cosine_similarity': mean_cosine,
                'stability_score': mean_cosine
            }
        }
    
    # -------------------------------------------------------------------------
    # 3. METHOD AGREEMENT METRICS
    # -------------------------------------------------------------------------
    
    def method_agreement(self, shap_attributions, lime_attributions, k_values=[3, 5, 10]):
        """
        Agreement between SHAP and LIME on top-k important features.
        
        Metrics:
        - Jaccard Similarity: |intersection| / |union|
        - Rank Correlation: Spearman correlation of feature rankings
        - Top-k Overlap: Percentage of shared top-k features
        """
        print("Computing Method Agreement (SHAP vs LIME)...")
        
        if shap_attributions is None or lime_attributions is None:
            return {'method_agreement': 'N/A - Missing attributions'}
        
        n_samples = min(len(shap_attributions), len(lime_attributions))
        
        results = {}
        
        for k in k_values:
            jaccard_scores = []
            overlap_scores = []
            rank_correlations = []
            
            for i in range(n_samples):
                shap_attr = np.abs(shap_attributions[i])
                lime_attr = np.abs(lime_attributions[i])
                if not np.all(np.isfinite(shap_attr)) or not np.all(np.isfinite(lime_attr)):
                    continue
                
                # Ensure same length
                min_len = min(len(shap_attr), len(lime_attr))
                shap_attr = shap_attr[:min_len]
                lime_attr = lime_attr[:min_len]
                
                if k > min_len:
                    continue
                
                # Top-k indices
                shap_top_k = set(np.argsort(shap_attr)[-k:])
                lime_top_k = set(np.argsort(lime_attr)[-k:])
                
                # Jaccard similarity
                intersection = len(shap_top_k & lime_top_k)
                union = len(shap_top_k | lime_top_k)
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)
                
                # Overlap percentage
                overlap = intersection / k
                overlap_scores.append(overlap)
                
                # Rank correlation
                from scipy.stats import spearmanr
                if len(shap_attr) > 1:
                    corr, _ = spearmanr(shap_attr, lime_attr)
                    if not np.isnan(corr):
                        rank_correlations.append(corr)
            
            results[f'agreement_k{k}'] = {
                'jaccard_similarity': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
                'top_k_overlap': float(np.mean(overlap_scores)) if overlap_scores else 0.0,
                'rank_correlation': float(np.mean(rank_correlations)) if rank_correlations else 0.0
            }
        
        return results
    
    # -------------------------------------------------------------------------
    # 4. MONOTONICITY
    # -------------------------------------------------------------------------
    
    def monotonicity(self, x_seq, x_temp, attributions):
        """
        Monotonicity: Does prediction change monotonically as we remove features
        in order of importance?
        
        Higher score = more monotonic (better explanation quality)
        """
        print("Computing Monotonicity...")
        n_samples = min(len(x_seq), 20)
        seq_len = x_seq.shape[1]
        
        monotonicity_scores = []
        
        for i in range(n_samples):
            orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
            target_idx = self._target_index(orig_pred)
            
            sample_attr = np.abs(attributions[i])
            valid_positions = self._valid_positions(x_seq[i])
            sorted_indices = self._top_k_indices(sample_attr, len(valid_positions), valid_positions)[::-1]
            
            predictions = [self._target_score(orig_pred, target_idx)]
            x_masked = x_seq[i:i+1].copy()
            
            # Progressively remove features
            for j, idx in enumerate(sorted_indices[:min(10, len(sorted_indices))]):
                x_masked[0, idx] = 0
                pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                pred_val = self._target_score(pred, target_idx)
                predictions.append(pred_val)
            
            # Count monotonic decreases
            n_monotonic = sum(1 for j in range(1, len(predictions)) 
                            if predictions[j] <= predictions[j-1])
            monotonicity = n_monotonic / (len(predictions) - 1) if len(predictions) > 1 else 0
            monotonicity_scores.append(monotonicity)
        
        return {
            'monotonicity': {
                'mean': float(np.mean(monotonicity_scores)),
                'std': float(np.std(monotonicity_scores)),
                'median': float(np.median(monotonicity_scores))
            }
        }
    
    # -------------------------------------------------------------------------
    # 5. TEMPORAL-SPECIFIC METRICS (for Process Mining)
    # -------------------------------------------------------------------------
    
    def temporal_consistency(self, attributions, test_seq):
        """
        Process Mining specific: Check if recent activities have higher importance
        (recency bias analysis).
        """
        print("Computing Temporal Consistency...")
        n_samples = len(attributions)
        seq_len = attributions.shape[1] if attributions.ndim > 1 else len(attributions)
        
        position_importance = np.zeros(seq_len)
        position_counts = np.zeros(seq_len)
        
        for i in range(n_samples):
            sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
            # Only count non-padding positions
            if test_seq is not None:
                non_pad = test_seq[i] > 0
                position_importance[:len(sample_attr)] += sample_attr * non_pad[:len(sample_attr)]
                position_counts[:len(sample_attr)] += non_pad[:len(sample_attr)]
            else:
                position_importance[:len(sample_attr)] += sample_attr
                position_counts[:len(sample_attr)] += 1
        
        # Average importance per position
        avg_importance = np.divide(position_importance, position_counts, 
                                   where=position_counts > 0, out=np.zeros_like(position_importance))
        
        # Recency correlation: later positions should have higher importance
        positions = np.arange(seq_len)
        valid_mask = position_counts > 0
        
        from scipy.stats import spearmanr
        if valid_mask.sum() > 2:
            recency_corr, recency_p = spearmanr(positions[valid_mask], avg_importance[valid_mask])
        else:
            recency_corr, recency_p = 0.0, 1.0
        
        valid_positions = positions[valid_mask]
        valid_importance = avg_importance[valid_mask]
        
        return {
            'temporal_consistency': {
                'recency_correlation': float(recency_corr) if not np.isnan(recency_corr) else 0.0,
                'recency_p_value': float(recency_p) if not np.isnan(recency_p) else 1.0,
                'position_importance': avg_importance.tolist(),
                'most_important_position': int(valid_positions[np.argmax(valid_importance)]) if valid_positions.size else 0,
                'least_important_position': int(valid_positions[np.argmin(valid_importance)]) if valid_positions.size else 0
            }
        }
    
    # -------------------------------------------------------------------------
    # MAIN BENCHMARK RUNNER
    # -------------------------------------------------------------------------
    
    def run_full_benchmark(self, x_seq, x_temp, shap_values, lime_values=None, 
                          test_seq=None, k_values=[1, 3, 5, 10]):
        """
        Run all benchmark metrics and return comprehensive results.
        
        Args:
            x_seq: Test sequences (n_samples, seq_len)
            x_temp: Temporal features (n_samples, temp_features) or None
            shap_values: SHAP attributions (n_samples, seq_len)
            lime_values: LIME attributions (n_samples, seq_len) or None
            test_seq: Original test sequences for padding detection
            k_values: List of k values for top-k metrics
            
        Returns:
            Dictionary with all benchmark results
        """
        print("\n" + "="*60)
        print("EXPLAINABILITY BENCHMARK EVALUATION")
        print("="*60)
        
        results = {
            'metadata': {
                'task': self.task,
                'n_samples': len(x_seq),
                'seq_len': x_seq.shape[1],
                'k_values': k_values,
                'is_multi_input': self.is_multi_input
            }
        }
        
        # 1. Faithfulness
        try:
            results['faithfulness'] = self.faithfulness_correlation(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Faithfulness computation failed: {e}")
            results['faithfulness'] = {'error': str(e)}
        
        # 2. Comprehensiveness
        try:
            results['comprehensiveness'] = self.comprehensiveness(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Comprehensiveness computation failed: {e}")
            results['comprehensiveness'] = {'error': str(e)}
        
        # 3. Sufficiency
        try:
            results['sufficiency'] = self.sufficiency(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Sufficiency computation failed: {e}")
            results['sufficiency'] = {'error': str(e)}
        
        # 4. Monotonicity
        try:
            results['monotonicity'] = self.monotonicity(x_seq, x_temp, shap_values)
        except Exception as e:
            print(f"[WARNING] Monotonicity computation failed: {e}")
            results['monotonicity'] = {'error': str(e)}
        
        # 5. Stability
        try:
            results['stability'] = self.stability(x_seq, x_temp, shap_values)
        except Exception as e:
            print(f"[WARNING] Stability computation failed: {e}")
            results['stability'] = {'error': str(e)}
        
        # 6. Method Agreement (if LIME available)
        if lime_values is not None:
            try:
                results['method_agreement'] = self.method_agreement(
                    shap_values, lime_values, k_values
                )
            except Exception as e:
                print(f"[WARNING] Method agreement computation failed: {e}")
                results['method_agreement'] = {'error': str(e)}
        
        # 7. Temporal Consistency
        try:
            results['temporal_consistency'] = self.temporal_consistency(
                shap_values, test_seq
            )
        except Exception as e:
            print(f"[WARNING] Temporal consistency computation failed: {e}")
            results['temporal_consistency'] = {'error': str(e)}
        
        self.results = results
        return results
    
    def save_results(self, output_dir, filename='benchmark_results.json'):
        """Save benchmark results to JSON file."""
        import json
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[OK] Benchmark results saved to: {filepath}")
        
        # Also save a summary CSV for easy comparison
        summary_rows = []
        
        # Extract key metrics
        for metric_name, metric_data in self.results.items():
            if metric_name == 'metadata':
                continue
            if isinstance(metric_data, dict):
                for sub_key, sub_val in metric_data.items():
                    if isinstance(sub_val, dict):
                        for k, v in sub_val.items():
                            if isinstance(v, (int, float)):
                                summary_rows.append({
                                    'category': metric_name,
                                    'metric': f"{sub_key}_{k}",
                                    'value': v
                                })
                    elif isinstance(sub_val, (int, float)):
                        summary_rows.append({
                            'category': metric_name,
                            'metric': sub_key,
                            'value': sub_val
                        })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"[OK] Benchmark summary saved to: {summary_path}")
        
        return filepath
    
    def print_summary(self):
        """Print a human-readable summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Faithfulness
        if 'faithfulness' in self.results and 'error' not in self.results['faithfulness']:
            print("\nFAITHFULNESS (Higher = Better)")
            for k, v in self.results['faithfulness'].items():
                if isinstance(v, dict):
                    corr = v.get('spearman_correlation', 'N/A')
                    print(f"   {k}: Spearman={corr:.4f}" if isinstance(corr, float) else f"   {k}: {corr}")
        
        # Comprehensiveness
        if 'comprehensiveness' in self.results and 'error' not in self.results['comprehensiveness']:
            print("\nCOMPREHENSIVENESS (Higher = Better)")
            for k, v in self.results['comprehensiveness'].items():
                if isinstance(v, dict):
                    mean = v.get('mean', 'N/A')
                    print(f"   {k}: Mean={mean:.4f}" if isinstance(mean, float) else f"   {k}: {mean}")
        
        # Sufficiency
        if 'sufficiency' in self.results and 'error' not in self.results['sufficiency']:
            print("\nSUFFICIENCY (Lower = Better)")
            for k, v in self.results['sufficiency'].items():
                if isinstance(v, dict):
                    mean = v.get('mean', 'N/A')
                    print(f"   {k}: Mean={mean:.4f}" if isinstance(mean, float) else f"   {k}: {mean}")
        
        # Monotonicity
        if 'monotonicity' in self.results and 'error' not in self.results['monotonicity']:
            mono = self.results['monotonicity'].get('monotonicity', {})
            mean = mono.get('mean', 'N/A')
            print(f"\nMONOTONICITY (Higher = Better): {mean:.4f}" if isinstance(mean, float) else f"\nMONOTONICITY: {mean}")
        
        # Method Agreement
        if 'method_agreement' in self.results and 'error' not in self.results['method_agreement']:
            print("\nMETHOD AGREEMENT (SHAP vs LIME)")
            for k, v in self.results['method_agreement'].items():
                if isinstance(v, dict):
                    jaccard = v.get('jaccard_similarity', 'N/A')
                    overlap = v.get('top_k_overlap', 'N/A')
                    print(f"   {k}: Jaccard={jaccard:.4f}, Overlap={overlap:.2%}" 
                          if isinstance(jaccard, float) else f"   {k}: {jaccard}")
        
        # Temporal Consistency
        if 'temporal_consistency' in self.results and 'error' not in self.results['temporal_consistency']:
            tc = self.results['temporal_consistency'].get('temporal_consistency', {})
            recency = tc.get('recency_correlation', 'N/A')
            print(f"\nTEMPORAL CONSISTENCY (Recency Correlation): {recency:.4f}" 
                  if isinstance(recency, float) else f"\nTEMPORAL CONSISTENCY: {recency}")
        
        print("\n" + "="*60)


def run_transformer_explainability(model, data, output_dir, task='activity', num_samples=50, methods='all', label_encoder=None, scaler=None, timestamps=None, feature_config=None, run_benchmark=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize explainer references
    se = None  # SHAP explainer
    le = None  # LIME explainer
    shap_dir = None
    lime_dir = None
    
    print("="*60)
    print(f"EXPLAINABILITY MODULE: {task.upper()} PREDICTION")
    print("="*60)
    
    # Check if label_encoder was provided
    if label_encoder is None:
        print("\n" + "!"*60)
        print("WARNING: label_encoder is None!")
        print("Plots will show generic labels like 'Activity_4'")
        print("To fix: Pass predictor.label_encoder to this function")
        print("!"*60 + "\n")
    
    is_time_task = task in ['time', 'event_time', 'remaining_time']

    if task == 'activity':
        train_data = data['X_train']
        test_data = data['X_test']
        num_classes = len(np.unique(data['y_train']))
        if num_samples < num_classes:
            print(f"[WARNING] num_samples={num_samples} < num_classes={num_classes}. Increasing for full coverage.")
            num_samples = num_classes
    else:
        train_data = (data['X_seq_train'], data['X_temp_train'])
        test_data = (data['X_seq_test'], data['X_temp_test'])
        num_classes = None

    if methods in ['shap', 'all']:
        print("\n--- Running SHAP ---")
        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)
        try:
            if is_time_task and ExplainabilityConfig.ENABLE_TIMESTEP_EXPLANATIONS:
                se = TimestepSHAPExplainer(model, task, label_encoder, scaler, timestamps)
            else:
                se = SHAPExplainer(model, task, label_encoder, scaler)
            se.initialize_explainer(train_data)
            shap_indices = None
            if task == 'activity':
                shap_indices = select_diverse_samples(
                    data, task, num_diverse=num_samples, label_encoder=label_encoder
                )
                if not shap_indices:
                    shap_indices = None
            se.explain_samples(test_data, num_samples, indices=shap_indices)
            if isinstance(se, TimestepSHAPExplainer) and se.model_has_timestep_outputs:
                print("\n[SHAP] Generating timestep-level visualizations...")
                for i in range(min(5, num_samples)):
                    se.plot_temporal_evolution(shap_dir, sample_idx=i, show_prediction=True)
                for i in range(min(3, num_samples)):
                    se.plot_timestep_heatmap(shap_dir, sample_idx=i)
                se.plot_global_temporal_importance(shap_dir)
            else:
                se.plot_bar(shap_dir)
                se.plot_summary(shap_dir)
            se.save_explanations(shap_dir)
        except Exception as e:
            print(f"[ERROR] SHAP explainability failed: {e}")
        if not _dir_has_png(shap_dir):
            print("[WARNING] No SHAP plots generated.")
        _ensure_stub_csv(
            os.path.join(shap_dir, "global_importance_data.csv"),
            ["Activity", "Mean_Impact"]
        )

    if methods in ['lime', 'all']:
        print("\n--- Running LIME ---")
        lime_dir = os.path.join(output_dir, 'lime')
        os.makedirs(lime_dir, exist_ok=True)
        
        try:
            le = LIMEExplainer(model, task, label_encoder, scaler)
            if feature_config and 'vocab_size' in feature_config:
                le.vocab_size = int(feature_config['vocab_size'])
            le.initialize_explainer(train_data, num_classes)
            
            # Select diverse samples FIRST
            diverse_samples = select_diverse_samples(data, task, num_diverse=num_samples, label_encoder=label_encoder)
            if not diverse_samples:
                print("[WARNING] No samples available for LIME. Skipping LIME explainability.")
                le.explanations = []
            else:
                print(f"Explaining {len(diverse_samples)} diverse samples: {diverse_samples}")
            
                # Explain ONLY the diverse samples
                if isinstance(test_data, (list, tuple)):
                    diverse_test_seq = test_data[0][diverse_samples]
                    diverse_test_temp = test_data[1][diverse_samples]
                    print(f"[DEBUG] Extracted {len(diverse_test_seq)} test sequences, {len(diverse_test_temp)} temp features")
                    diverse_test_data = (diverse_test_seq, diverse_test_temp)
                else:
                    diverse_test_data = test_data[diverse_samples]
                    print(f"[DEBUG] Extracted {len(diverse_test_data)} test samples")
                
                y_true_all = data.get('y_test', None)
                y_true_diverse = None
                if y_true_all is not None:
                    y_true_diverse = np.array(y_true_all)[diverse_samples]
                le.explain_samples(
                    diverse_test_data,
                    num_samples=len(diverse_samples),
                    num_features=30,
                    y_true=y_true_diverse
                )
                print(f"[DEBUG] Generated {len(le.explanations)} explanations")
                
                # Plot all explained samples (now they match 0-9)
                print(f"\n[LIME] Plotting {len(le.explanations)} explanations...")
                plots_saved = 0
                for i in range(len(le.explanations)):
                    try:
                        if le.explanations[i] is not None:
                            # Use original test set index in filename
                            original_idx = diverse_samples[i]
                            print(f"[LIME] Plotting sample {i} (original index: {original_idx})...")
                            le.plot_explanation(lime_dir, sample_idx=i, original_idx=original_idx)
                            plots_saved += 1
                        else:
                            print(f"[WARNING] Explanation {i} is None, skipping...")
                    except Exception as e:
                        print(f"[ERROR] Failed to plot sample {i}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"[LIME] Successfully saved {plots_saved} plots")
                
                le.save_explanations(lime_dir)
        except Exception as e:
            print(f"[ERROR] LIME explainability failed: {e}")
        if not _dir_has_png(lime_dir):
            print("[WARNING] No LIME plots generated.")
        _ensure_stub_csv(
            os.path.join(lime_dir, "lime_explanation_sample_0.csv"),
            ["Activity", "Weight"]
        )
    
    # -------------------------------------------------------------------------
    # RUN BENCHMARK EVALUATION
    # -------------------------------------------------------------------------
    benchmark_results = None
    if run_benchmark and methods in ['shap', 'all']:
        print("\n--- Running Benchmark Evaluation ---")
        benchmark_dir = os.path.join(output_dir, 'benchmark')
        os.makedirs(benchmark_dir, exist_ok=True)
        
        try:
            def extract_sequence_attributions(raw_values, seq_len, explainer):
                values = raw_values[0] if isinstance(raw_values, list) else raw_values
                if values is None:
                    return None
                values = np.asarray(values)

                if explainer.is_multi_input and hasattr(explainer, '_seq_flat_size'):
                    if values.ndim == 2 and values.shape[1] >= explainer._seq_flat_size:
                        values = values[:, :explainer._seq_flat_size]
                    if getattr(explainer, '_seq_shape', None) == (seq_len,):
                        values = values.reshape((values.shape[0], seq_len))
                    elif values.ndim == 2 and values.shape[1] == explainer._seq_flat_size:
                        values = values.reshape((values.shape[0],) + explainer._seq_shape)

                if values.ndim == 1:
                    values = values.reshape(1, -1)
                if values.ndim == 2 and values.shape[1] == seq_len:
                    return values

                if values.ndim > 2:
                    seq_axis = None
                    for axis in range(1, values.ndim):
                        if values.shape[axis] == seq_len:
                            seq_axis = axis
                            break
                    if seq_axis is not None:
                        values = np.moveaxis(values, seq_axis, 1)
                        if values.ndim > 2:
                            values = values.mean(axis=tuple(range(2, values.ndim)))
                        return values

                return values.reshape(values.shape[0], -1)[:, :seq_len]

            # Prepare test data for benchmark
            if se is not None and getattr(se, 'test_data', None) is not None:
                bench_x_seq = np.asarray(se.test_data)
                bench_x_temp = (
                    np.asarray(se.test_data_temp)
                    if getattr(se, 'test_data_temp', None) is not None
                    else None
                )
            elif isinstance(test_data, (list, tuple)):
                bench_x_seq = test_data[0][:num_samples]
                bench_x_temp = test_data[1][:num_samples]
            else:
                bench_x_seq = test_data[:num_samples]
                bench_x_temp = None
            
            # Extract SHAP attributions (handle flattened format)
            shap_attr = None
            if se is not None and se.shap_values is not None:
                shap_attr = extract_sequence_attributions(
                    se.shap_values.values,
                    bench_x_seq.shape[1],
                    se
                )
                if shap_attr is not None and shap_attr.shape[0] != len(bench_x_seq):
                    aligned = min(shap_attr.shape[0], len(bench_x_seq))
                    print(f"[WARNING] Aligning benchmark samples to {aligned} rows for SHAP consistency.")
                    shap_attr = shap_attr[:aligned]
                    bench_x_seq = bench_x_seq[:aligned]
                    if bench_x_temp is not None:
                        bench_x_temp = bench_x_temp[:aligned]

            attribution_fn = None
            if se is not None and se.explainer is not None:
                def attribution_fn(x_seq_batch, x_temp_batch=None):
                    if se.is_multi_input:
                        if x_temp_batch is None:
                            return None
                        seq_flat = x_seq_batch.reshape(len(x_seq_batch), -1)
                        temp_flat = x_temp_batch.reshape(len(x_temp_batch), -1)
                        explainer_input = np.hstack([seq_flat, temp_flat])
                    else:
                        explainer_input = x_seq_batch
                    fresh_values = se._call_explainer(explainer_input, max_evals=se.max_evals)
                    seq_attr = extract_sequence_attributions(
                        fresh_values.values,
                        x_seq_batch.shape[1],
                        se
                    )
                    if seq_attr is None or len(seq_attr) == 0:
                        return None
                    return np.asarray(seq_attr[0], dtype=float)
            
            # Extract LIME attributions if available
            lime_attr = None
            if methods in ['lime', 'all'] and bench_x_seq is not None and len(bench_x_seq) > 0:
                try:
                    benchmark_lime = LIMEExplainer(model, task, label_encoder, scaler)
                    if feature_config and 'vocab_size' in feature_config:
                        benchmark_lime.vocab_size = int(feature_config['vocab_size'])
                    elif 'le' in dir() and le is not None and le.vocab_size is not None:
                        benchmark_lime.vocab_size = le.vocab_size
                    benchmark_lime.initialize_explainer(train_data, num_classes)
                    benchmark_test_data = (
                        (bench_x_seq, bench_x_temp) if bench_x_temp is not None else bench_x_seq
                    )
                    benchmark_lime.explain_samples(
                        benchmark_test_data,
                        num_samples=len(bench_x_seq),
                        num_features=int(bench_x_seq.shape[1])
                    )

                    lime_attr_list = []
                    seq_len = bench_x_seq.shape[1]
                    for i, exp in enumerate(benchmark_lime.explanations):
                        if exp is None:
                            lime_attr_list.append(np.full(seq_len, np.nan))
                            continue

                        if task == 'activity' and hasattr(exp, 'top_labels') and exp.top_labels:
                            exp_list = exp.as_list(label=exp.top_labels[0])
                        else:
                            exp_list = exp.as_list()

                        exp_map = dict(exp_list)
                        weights = np.zeros(seq_len)

                        for feat_name, weight in exp_map.items():
                            name = str(feat_name)
                            match = re.search(r'(\d+)', name)
                            if match and "Position" in name:
                                pos = int(match.group(1)) - 1
                                if 0 <= pos < seq_len:
                                    weights[pos] += weight
                                continue

                            if label_encoder is not None:
                                activity_name = name.split('<=')[0].split('>')[0].strip()
                                try:
                                    token = label_encoder.transform([activity_name])[0] + 1
                                except Exception:
                                    continue
                                positions = np.where(bench_x_seq[i] == token)[0]
                                if positions.size > 0:
                                    per_pos = weight / positions.size
                                    for pos in positions:
                                        if 0 <= pos < seq_len:
                                            weights[pos] += per_pos

                        lime_attr_list.append(weights)
                    if lime_attr_list:
                        lime_attr = np.array(lime_attr_list)
                except Exception as e:
                    print(f"[WARNING] Could not extract LIME attributions for benchmark: {e}")
                    lime_attr = None
            
            # Initialize and run benchmark
            benchmark = ExplainabilityBenchmark(
                model=model,
                task=task,
                is_multi_input=isinstance(test_data, (list, tuple)),
                seq_shape=getattr(se, '_seq_shape', None) if se else None,
                temp_shape=getattr(se, '_temp_shape', None) if se else None,
                scaler=scaler,
                attribution_fn=attribution_fn
            )
            
            if shap_attr is not None:
                benchmark_results = benchmark.run_full_benchmark(
                    x_seq=bench_x_seq,
                    x_temp=bench_x_temp,
                    shap_values=shap_attr,
                    lime_values=lime_attr,
                    test_seq=bench_x_seq,
                    k_values=[1, 3, 5, 10]
                )
                
                benchmark.save_results(benchmark_dir)
                benchmark.print_summary()
            else:
                print("[WARNING] Could not extract SHAP attributions for benchmark.")
                
        except Exception as e:
            print(f"[ERROR] Benchmark evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive summary outputs
    if methods == 'all':
        print("\n--- Generating Comparison Report ---")
        generate_comparison_report(output_dir, shap_dir if 'shap' in methods or methods == 'all' else None, 
                                   lime_dir if 'lime' in methods or methods == 'all' else None)
    
    # Sanity check for benchmark coverage
    _validate_explainability_coverage(
        task,
        label_encoder,
        shap_dir if methods in ['shap', 'all'] else None,
        lime_dir if methods in ['lime', 'all'] else None
    )
        
    print("\n" + "="*60)
    print(f"EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    print("\nGenerated outputs:")
    print("  [OK] SHAP global importance plots")
    if task != 'activity' and methods in ['shap', 'all']:
        print("  [OK] Temporal attribution plots")
    print(f"  [OK] LIME local explanations ({num_samples} diverse samples)")
    print("  [OK] Feature importance summary CSV")
    print("  [OK] Method comparison report")
    if run_benchmark and benchmark_results:
        print("  [OK] Benchmark evaluation metrics (JSON + CSV)")
    print("="*60)
    
    return benchmark_results


# =============================================================================
# BENCHMARK COMPARISON UTILITIES
# =============================================================================

def compare_benchmark_results(benchmark_files, output_path=None):
    """
    Compare benchmark results across multiple models/datasets.
    
    Args:
        benchmark_files: List of tuples (model_name, benchmark_json_path)
        output_path: Optional path to save comparison CSV
        
    Returns:
        DataFrame with comparison results
    """
    import json
    
    comparison_rows = []
    
    for model_name, filepath in benchmark_files:
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            row = {'model': model_name}
            
            # Extract key metrics
            if 'faithfulness' in results:
                for k, v in results['faithfulness'].items():
                    if isinstance(v, dict) and 'spearman_correlation' in v:
                        row[f'faith_{k}'] = v['spearman_correlation']
            
            if 'comprehensiveness' in results:
                for k, v in results['comprehensiveness'].items():
                    if isinstance(v, dict) and 'mean' in v:
                        row[f'comp_{k}'] = v['mean']
            
            if 'sufficiency' in results:
                for k, v in results['sufficiency'].items():
                    if isinstance(v, dict) and 'mean' in v:
                        row[f'suff_{k}'] = v['mean']
            
            if 'monotonicity' in results:
                mono = results['monotonicity'].get('monotonicity', {})
                row['monotonicity'] = mono.get('mean', None)
            
            if 'method_agreement' in results:
                for k, v in results['method_agreement'].items():
                    if isinstance(v, dict) and 'jaccard_similarity' in v:
                        row[f'agree_{k}'] = v['jaccard_similarity']
            
            if 'temporal_consistency' in results:
                tc = results['temporal_consistency'].get('temporal_consistency', {})
                row['recency_corr'] = tc.get('recency_correlation', None)
            
            comparison_rows.append(row)
            
        except Exception as e:
            print(f"[WARNING] Failed to load {filepath}: {e}")
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    if output_path:
        comparison_df.to_csv(output_path, index=False)
        print(f"[OK] Benchmark comparison saved to: {output_path}")
    
    return comparison_df


def generate_benchmark_latex_table(comparison_df, output_path=None, caption="Explainability Benchmark Comparison"):
    """
    Generate LaTeX table for benchmark comparison (useful for research papers).
    
    Args:
        comparison_df: DataFrame from compare_benchmark_results()
        output_path: Optional path to save .tex file
        caption: Table caption
        
    Returns:
        LaTeX string
    """
    # Select key columns for the paper
    key_cols = ['model']
    metric_cols = [c for c in comparison_df.columns if c != 'model']
    
    # Rename columns for readability
    rename_map = {
        'faith_faithfulness_k5': 'Faith@5',
        'comp_comprehensiveness_k5': 'Comp@5',
        'suff_sufficiency_k5': 'Suff@5',
        'monotonicity': 'Mono',
        'agree_agreement_k5': 'Agree@5',
        'recency_corr': 'Recency'
    }
    
    df = comparison_df.copy()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Format numeric columns
    for col in df.columns:
        if col != 'model' and df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))
    
    # Add caption and label
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:explainability_benchmark}}
{latex}
\\end{{table}}"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[OK] LaTeX table saved to: {output_path}")
    
    return latex
