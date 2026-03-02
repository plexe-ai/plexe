"""
Standard Keras predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: keras, scikit-learn, pandas, cloudpickle.
"""

import json
import logging
import os
from pathlib import Path

# CRITICAL: Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import cloudpickle
import pandas as pd

logger = logging.getLogger(__name__)


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

        self.task_type = None
        metadata_path = artifacts_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.task_type = json.load(f).get("task_type")

    @staticmethod
    def _is_classification_task(task_type: str | None) -> bool:
        # TODO(task-type-enum): Switch this helper to the canonical TaskType enum once merged.
        if not task_type:
            return False
        normalized = str(task_type).strip().lower()
        return normalized in {"classification", "binary_classification", "multiclass_classification"}

    def _ensure_classification_for_proba(self):
        task_type = getattr(self, "task_type", None)
        if self._is_classification_task(task_type):
            return
        raise NotImplementedError(
            f"{type(self).__name__}.predict_proba() is only available for classification models. "
            f"Detected task_type='{task_type}'."
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

        # For classification: argmax to get class
        if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
            # Multi-class classification
            predictions = np.argmax(raw_predictions, axis=1)
        else:
            # Binary classification or regression
            if raw_predictions.shape[-1] == 1:
                # Single output - squeeze to 1D
                predictions = raw_predictions.squeeze()
                # For binary classification, threshold at 0.5
                if predictions.max() <= 1.0 and predictions.min() >= 0.0:
                    predictions = (predictions > 0.5).astype(int)
            else:
                predictions = raw_predictions

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Returns raw model outputs (sigmoid/softmax values) without argmax.
        """
        import numpy as np

        self._ensure_classification_for_proba()

        x_transformed = self.pipeline.transform(x)
        raw_predictions = self.model.predict(x_transformed, verbose=0)

        probabilities = np.asarray(raw_predictions)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        if probabilities.shape[1] == 1:
            probabilities = np.column_stack([1 - probabilities[:, 0], probabilities[:, 0]])

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
