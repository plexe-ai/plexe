"""
Training runner abstract base class.

Plugin interface for different training execution environments.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline


class TrainingRunner(ABC):
    """
    Abstract base class for training execution environments.

    Implementations can run training locally, on SageMaker, EMR, etc.
    """

    @abstractmethod
    def run_training(
        self,
        template: str,
        model: Any,
        feature_pipeline: Pipeline,
        train_uri: str,
        val_uri: str,
        timeout: int,
        target_columns: list[str],
    ) -> Path:
        """
        Execute model training and return path to artifacts.

        Args:
            template: Template name ("train_xgboost" or "train_pytorch")
            model: Untrained model object (XGBClassifier/XGBRegressor or nn.Module)
            feature_pipeline: Fitted sklearn Pipeline for feature transformations
            train_uri: URI to training data
            val_uri: URI to validation data
            timeout: Maximum training time (seconds)
            target_columns: List of target column names

        Returns:
            Path to directory containing:
                - model.json/model.pt (trained model in standard format)
                - pipeline.pkl (feature pipeline)
                - metadata.json (training metadata)

        Raises:
            TrainingError: If training fails
        """
        pass
