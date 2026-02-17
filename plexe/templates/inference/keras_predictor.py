"""
Standard Keras predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: keras, scikit-learn, pandas, cloudpickle.
"""

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
