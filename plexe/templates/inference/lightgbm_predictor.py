"""
Standard LightGBM predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: lightgbm, scikit-learn, pandas.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


class LightGBMPredictor:
    """
    Standalone LightGBM predictor.

    No custom dependencies - standard libraries only.
    """

    def __init__(self, model_dir: str):
        """
        Load model from directory.

        Args:
            model_dir: Path to model package directory
        """
        model_dir = Path(model_dir)
        artifacts_dir = model_dir / "artifacts"

        metadata_path = artifacts_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self._task_type = metadata.get("task_type", "")
        else:
            self._task_type = ""

        # Execute pipeline code (defines custom FunctionTransformer functions)
        code_path = model_dir / "src" / "pipeline.py"
        if code_path.exists():
            with open(code_path) as f:
                exec(f.read(), globals())

        # Load model artifacts
        self.model = joblib.load(artifacts_dir / "model.pkl")
        self.pipeline = joblib.load(artifacts_dir / "pipeline.pkl")

        # Load label encoder (for classification with non-contiguous labels)
        encoder_path = artifacts_dir / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input DataFrame.

        Args:
            x: Input features DataFrame (assumes correct dtypes)

        Returns:
            DataFrame with predictions
        """
        # Apply feature pipeline and make prediction
        predictions = self.model.predict(self.pipeline.transform(x))

        # Decode labels if encoder exists
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Returns:
            DataFrame with probability columns named proba_0..proba_n.
        """
        if self._task_type and self._task_type not in {"binary_classification", "multiclass_classification"}:
            raise ValueError(
                f"predict_proba() is only valid for classification tasks, got task_type='{self._task_type or 'unknown'}'"
            )
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(f"Model type {type(self.model).__name__} does not support predict_proba().")

        probabilities = np.asarray(self.model.predict_proba(self.pipeline.transform(x)))
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
    predictor = LightGBMPredictor(model_dir="./model")

    sample_input = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
        }
    )

    predictions = predictor.predict(sample_input)
    print(predictions)
