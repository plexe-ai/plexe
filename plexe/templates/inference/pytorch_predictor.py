"""
Standard PyTorch predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: torch, scikit-learn, pandas, cloudpickle.
"""

import json
import logging
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import scipy.sparse
import torch

logger = logging.getLogger(__name__)


class PyTorchPredictor:
    """
    Standalone PyTorch predictor.

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

        # Load full model via cloudpickle (includes class definition + trained weights)
        model_class_path = artifacts_dir / "model_class.pkl"
        with open(model_class_path, "rb") as f:
            self.model = cloudpickle.load(f)

        # Load trained state dict and apply
        state_dict_path = artifacts_dir / "model.pt"
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location="cpu"))
        self.model.eval()

        # Load feature pipeline
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
        # Transform features through pipeline
        x_transformed = self.pipeline.transform(x)

        # Handle sparse matrix output (e.g. from OneHotEncoder, CountVectorizer)
        if scipy.sparse.issparse(x_transformed):
            x_transformed = x_transformed.toarray()

        x_tensor = torch.tensor(np.array(x_transformed, dtype=np.float32))

        # Predict
        with torch.no_grad():
            raw_output = self.model(x_tensor)

        raw_predictions = raw_output.detach().cpu().numpy()

        # Post-process based on output shape
        if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
            # Multi-class classification: argmax
            predictions = np.argmax(raw_predictions, axis=1)
        else:
            # Single output: squeeze to 1D
            predictions = raw_predictions.squeeze()
            # Binary classification: threshold at 0.5 if output looks like probabilities
            # FIXME: This heuristic can misclassify regression outputs in [0, 1];
            # we'll address this in a follow-up PR with richer task metadata.
            if predictions.ndim > 0 and predictions.max() <= 1.0 and predictions.min() >= 0.0:
                predictions = (predictions > 0.5).astype(int)

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Applies sigmoid for single-logit binary models, otherwise softmax.
        """
        self._ensure_classification_for_proba()

        # Transform features through pipeline
        x_transformed = self.pipeline.transform(x)

        # Handle sparse matrix output (e.g. from OneHotEncoder, CountVectorizer)
        if scipy.sparse.issparse(x_transformed):
            x_transformed = x_transformed.toarray()

        x_tensor = torch.tensor(np.array(x_transformed, dtype=np.float32))

        with torch.no_grad():
            raw_output = self.model(x_tensor)

        raw_output = raw_output.detach().cpu()
        if raw_output.ndim == 1:
            raw_output = raw_output.unsqueeze(1)

        if raw_output.shape[1] == 1:
            proba_pos = torch.sigmoid(raw_output).squeeze(1)
            probabilities = torch.stack([1 - proba_pos, proba_pos], dim=1)
        else:
            probabilities = torch.softmax(raw_output, dim=1)

        probabilities = probabilities.numpy()
        columns = [f"proba_{i}" for i in range(probabilities.shape[1])]
        return pd.DataFrame(probabilities, columns=columns)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    predictor = PyTorchPredictor(model_dir="./model")

    sample_input = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
        }
    )

    predictions = predictor.predict(sample_input)
    print(predictions)
