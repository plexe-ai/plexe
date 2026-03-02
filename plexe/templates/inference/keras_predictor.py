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
        else:
            raw_task_type = ""

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
            # Keras outputs probabilities — threshold at 0.5
            predictions = raw_predictions.squeeze()
            predictions = (predictions > 0.5).astype(int)
        elif self._task_type == "multiclass_classification":
            predictions = np.argmax(raw_predictions, axis=1)
        else:
            # Regression or unknown: return raw values
            predictions = raw_predictions.squeeze()

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Returns raw model outputs (sigmoid/softmax values) without argmax.
        """
        import numpy as np

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
