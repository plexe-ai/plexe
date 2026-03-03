"""
Standard Keras predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: keras, scikit-learn, pandas, cloudpickle.
"""

import json
import os
from pathlib import Path

# CRITICAL: Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import cloudpickle
import pandas as pd


class KerasPredictor:
    """
    Standalone Keras predictor.

    No custom dependencies - standard libraries only.
    """

    def __init__(self, model_dir: str):
        """
        Load model from directory.

        Args:
            model_dir: Path to model package directory
        """

        import keras

        model_dir = Path(model_dir)
        artifacts_dir = model_dir / "artifacts"

        # Load metadata for task_type-driven post-processing
        metadata_path = artifacts_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            raw_task_type = metadata.get("task_type", "")
            self._loss_class = metadata.get("loss_class", "")
            self._loss_config = metadata.get("loss_config", {}) or {}
        else:
            raw_task_type = ""
            self._loss_class = ""
            self._loss_config = {}

        self._task_type = raw_task_type

        # Execute pipeline code (defines custom FunctionTransformer functions)
        code_path = model_dir / "src" / "pipeline.py"
        if code_path.exists():
            with open(code_path) as f:
                exec(f.read(), globals())

        # Load model artifacts
        self.model = keras.models.load_model(artifacts_dir / "model.keras")

        # Load feature pipeline (custom functions available if code was exec'd)
        with open(artifacts_dir / "pipeline.pkl", "rb") as f:
            self.pipeline = cloudpickle.load(f)

    def _uses_logits_output(self, task_type: str | None = None) -> bool:
        """Return True when model outputs are logits based on training loss metadata."""
        effective_task_type = task_type or self._task_type
        from_logits = self._loss_config.get("from_logits")
        if effective_task_type == "binary_classification" and self._loss_class == "BinaryCrossentropy":
            return bool(from_logits)
        if effective_task_type == "multiclass_classification" and self._loss_class in {
            "SparseCategoricalCrossentropy",
            "CategoricalCrossentropy",
        }:
            return bool(from_logits)
        return False

    def _probabilities_from_raw(self, raw_predictions):
        """Convert raw model outputs into probability arrays."""
        import numpy as np

        probabilities = np.asarray(raw_predictions)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        if not np.isfinite(probabilities).all():
            raise ValueError("Keras model outputs contain NaN/Inf values; cannot compute probabilities.")
        task_type = self._task_type
        if not task_type:
            task_type = "binary_classification" if probabilities.shape[1] <= 2 else "multiclass_classification"
        uses_logits = self._uses_logits_output(task_type)

        # Legacy model metadata may omit loss_config.from_logits.
        # If outputs are clearly outside probability bounds, treat them as logits.
        if (
            not uses_logits
            and not self._loss_config
            and task_type
            in {
                "binary_classification",
                "multiclass_classification",
            }
        ):
            if probabilities.min() < 0.0 or probabilities.max() > 1.0:
                uses_logits = True

        if task_type == "binary_classification":
            if probabilities.shape[1] == 1:
                positive = probabilities[:, 0]
                if uses_logits:
                    positive = 1.0 / (1.0 + np.exp(-positive))
                probabilities = np.column_stack([1 - positive, positive])
                return probabilities

            if probabilities.shape[1] == 2:
                if uses_logits:
                    shifted = probabilities - np.max(probabilities, axis=1, keepdims=True)
                    exp_values = np.exp(shifted)
                    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                return probabilities

            raise ValueError(f"Binary classification expects 1 or 2 outputs, got shape {probabilities.shape}")

        if task_type == "multiclass_classification":
            if uses_logits:
                shifted = probabilities - np.max(probabilities, axis=1, keepdims=True)
                exp_values = np.exp(shifted)
                probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return probabilities

        raise ValueError(
            f"predict_proba() is only valid for classification tasks, got task_type='{self._task_type or 'unknown'}'"
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input DataFrame.

        Args:
            x: Input features DataFrame (assumes correct dtypes)

        Returns:
            DataFrame with predictions
        """
        import numpy as np

        # Transform features through pipeline
        x_transformed = self.pipeline.transform(x)

        # Keras predict returns probabilities/values
        raw_predictions = self.model.predict(x_transformed, verbose=0)

        # Post-process based on task_type from metadata
        if self._task_type == "binary_classification":
            probabilities = self._probabilities_from_raw(raw_predictions)
            predictions = (probabilities[:, 1] > 0.5).astype(int)
        elif self._task_type == "multiclass_classification":
            probabilities = self._probabilities_from_raw(raw_predictions)
            predictions = np.argmax(probabilities, axis=1)
        else:
            # Regression or unknown: return raw values
            predictions = raw_predictions.squeeze()

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Returns per-class probabilities with columns proba_0..proba_n.
        """
        x_transformed = self.pipeline.transform(x)
        raw_predictions = self.model.predict(x_transformed, verbose=0)

        probabilities = self._probabilities_from_raw(raw_predictions)

        columns = [f"proba_{i}" for i in range(probabilities.shape[1])]
        return pd.DataFrame(probabilities, columns=columns)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example: Load and use predictor
    predictor = KerasPredictor(model_dir="./model")

    # Create sample input
    sample_input = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
        }
    )

    # Predict
    predictions = predictor.predict(sample_input)
    print(predictions)
