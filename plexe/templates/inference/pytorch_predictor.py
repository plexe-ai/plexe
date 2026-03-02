"""
Standard PyTorch predictor - NO Plexe dependencies.

This file is copied as-is into model artifacts.
Can be used standalone with just: torch, scikit-learn, pandas, cloudpickle.
"""

import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import scipy.sparse
import torch


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

        # Load full model via cloudpickle (includes class definition + trained weights)
        model_class_path = artifacts_dir / "model_class.pkl"
        with open(model_class_path, "rb") as f:
            self.model = cloudpickle.load(f)

        # Load trained state dict and apply (always on CPU for portable inference)
        state_dict_path = artifacts_dir / "model.pt"
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location="cpu"))
        self.model.eval()

        # Load feature pipeline
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

        # Post-process based on task_type from metadata
        if self._task_type == "binary_classification":
            # BCEWithLogitsLoss outputs raw logits — apply sigmoid + threshold
            predictions = raw_predictions.squeeze()
            sigmoid = 1.0 / (1.0 + np.exp(-predictions))
            predictions = (sigmoid > 0.5).astype(int)
        elif self._task_type == "multiclass_classification":
            predictions = np.argmax(raw_predictions, axis=1)
        else:
            # Regression or unknown: return raw values
            predictions = raw_predictions.squeeze()

        return pd.DataFrame({"prediction": predictions})

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict per-class probabilities on input DataFrame.

        Applies sigmoid for single-logit binary models, otherwise softmax.
        """
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
