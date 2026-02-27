"""
Standard CatBoost predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: catboost, scikit-learn, pandas.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

logger = logging.getLogger(__name__)


class CatBoostPredictor:
    """
    Standalone CatBoost predictor.

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

        # Execute pipeline code (defines custom FunctionTransformer functions)
        code_path = model_dir / "src" / "pipeline.py"
        if code_path.exists():
            with open(code_path) as f:
                exec(f.read(), globals())

        # Load CatBoost model using native format
        # Try classifier first, then regressor
        model_path = artifacts_dir / "model.cbm"
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))
        except Exception:
            self.model = CatBoostRegressor()
            self.model.load_model(str(model_path))

        # Load sklearn pipeline
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

    def _proba_columns(self, n_classes: int) -> list[str]:
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            classes = list(self.label_encoder.classes_)
        else:
            classes = list(range(n_classes))

        if len(classes) != n_classes:
            logger.warning(
                "Label encoder classes length (%s) does not match probability columns (%s); "
                "falling back to index labels.",
                len(classes),
                n_classes,
            )
            classes = list(range(n_classes))

        return [f"proba_{cls}" for cls in classes]

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Args:
            x: Input features DataFrame (assumes correct dtypes)

        Returns:
            DataFrame with per-class probability columns
        """
        import numpy as np

        probabilities = self.model.predict_proba(self.pipeline.transform(x))
        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])

        columns = self._proba_columns(probabilities.shape[1])
        return pd.DataFrame(probabilities, columns=columns)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example: Load and use predictor
    predictor = CatBoostPredictor(model_dir="./model")

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
