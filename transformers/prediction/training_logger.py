from tensorflow import keras


class EpochMetricsLogger(keras.callbacks.Callback):
    """Print a stable one-line metric summary after each epoch."""

    DISPLAY_ORDER = (
        "loss",
        "accuracy",
        "mae",
        "time_output_loss",
        "time_output_mae",
        "val_loss",
        "val_accuracy",
        "val_mae",
        "val_time_output_loss",
        "val_time_output_mae",
    )

    DISPLAY_NAMES = {
        "loss": "loss",
        "accuracy": "accuracy",
        "mae": "mae",
        "time_output_loss": "time_output_loss",
        "time_output_mae": "time_output_mae",
        "val_loss": "val_loss",
        "val_accuracy": "val_accuracy",
        "val_mae": "val_mae",
        "val_time_output_loss": "val_time_output_loss",
        "val_time_output_mae": "val_time_output_mae",
    }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total_epochs = self.params.get("epochs", "?")
        ordered_keys = [key for key in self.DISPLAY_ORDER if key in logs]
        extra_keys = sorted(
            key for key in logs.keys()
            if key not in ordered_keys and not key.startswith("lr")
        )

        parts = []
        for key in ordered_keys + extra_keys:
            value = logs.get(key)
            if value is None:
                continue
            name = self.DISPLAY_NAMES.get(key, key)
            if "accuracy" in key:
                parts.append(f"{name}: {float(value) * 100:.2f}%")
            else:
                parts.append(f"{name}: {float(value):.4f}")

        print(f"Epoch {epoch + 1}/{total_epochs} - " + " - ".join(parts), flush=True)
