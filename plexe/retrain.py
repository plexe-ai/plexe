"""
Model retraining functionality.

Retrain existing models with new data using original training pipeline.
"""

import inspect
import os
import json
import logging
import shutil
import tarfile
from pathlib import Path

import joblib
import pandas as pd

# Ensure keras uses TensorFlow backend even when retrain is invoked directly.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from plexe.constants import DirNames
from plexe.models import Solution
from plexe.templates.features.pipeline_fitter import fit_pipeline
from plexe.utils.s3 import download_s3_uri

logger = logging.getLogger(__name__)


class RetrainingError(Exception):
    """Raised when retraining fails."""

    pass


def retrain_model(
    original_model_uri: str,
    train_dataset_uri: str,
    experiment_id: str,
    work_dir: Path,
    runner,  # TrainingRunner instance
    config,  # Config instance
    on_checkpoint_saved=None,
) -> tuple[Solution, dict]:
    """
    Retrain existing model with new data using original training pipeline.

    Steps:
    1. Extract original model package
    2. Load pipeline and transform new data
    3. Run trainer.py (extracted from package)
    4. Copy original package + replace artifacts

    Args:
        original_model_uri: Local path to model.tar.gz (integration prepares this)
        train_dataset_uri: Path or S3 URI to training dataset (S3 URIs automatically downloaded)
        experiment_id: Experiment identifier
        work_dir: Working directory
        on_checkpoint_saved: Optional callback (same as build_model)

    Returns:
        (solution, final_metrics) tuple

    Raises:
        RetrainingError: If retraining fails
    """

    logger.info(f"Starting model retraining for experiment {experiment_id}")
    logger.info(f"Original model: {original_model_uri}")

    # Create working directories
    work_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = work_dir / "original_model"
    retrain_dir = work_dir / DirNames.BUILD_DIR / "retrain"  # Hidden intermediate files
    retrain_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ============================================
        # Step 1: Extract Original Model Package
        # ============================================
        logger.info("Extracting original model package...")
        _extract_package(original_model_uri, extract_dir)

        # Load model metadata
        metadata_path = extract_dir / "artifacts" / "metadata.json"
        if not metadata_path.exists():
            raise RetrainingError(f"Model metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        model_type = metadata.get("model_type")
        target_column = metadata.get("target_column")
        logger.info(f"Model type: {model_type}, target: {target_column}")

        # Verify trainer.py exists
        trainer_path = extract_dir / "src" / "trainer.py"
        if not trainer_path.exists():
            raise RetrainingError(
                f"Training code not found in package: {trainer_path}. "
                "This model was created before retraining support was added."
            )

        # ============================================
        # Step 2: Load Pipeline, Fit, and Transform New Data
        # ============================================
        logger.info("Loading and refitting feature pipeline...")

        # Execute pipeline code to get unfitted pipeline object
        pipeline_code_path = extract_dir / "src" / "pipeline.py"
        if not pipeline_code_path.exists():
            raise RetrainingError(f"Pipeline code not found: {pipeline_code_path}")

        pipeline_namespace = {}
        with open(pipeline_code_path) as f:
            exec(f.read(), pipeline_namespace)

        if "pipeline" not in pipeline_namespace:
            raise RetrainingError("Pipeline code must define a 'pipeline' variable")

        unfitted_pipeline = pipeline_namespace["pipeline"]
        logger.info("Loaded unfitted pipeline from src/pipeline.py")

        # Load new training data
        logger.info(f"Loading training data from {train_dataset_uri}...")
        if train_dataset_uri.startswith("s3://"):
            train_dataset_uri = download_s3_uri(train_dataset_uri)
        new_data = pd.read_parquet(train_dataset_uri)
        logger.info(f"Data shape: {new_data.shape}")

        # Validate target column exists
        if target_column not in new_data.columns:
            raise RetrainingError(
                f"Target column '{target_column}' not found in new data. "
                f"Available columns: {list(new_data.columns)}"
            )

        # Split raw data (80/20 train/val)
        logger.info("Splitting data (80/20 train/val)...")
        train_raw, val_raw = train_test_split(new_data, test_size=0.2, random_state=42)
        logger.info(f"Raw splits - Train: {train_raw.shape}, Val: {val_raw.shape}")

        # Save raw train data temporarily for fit_pipeline utility
        train_raw_path = retrain_dir / "train_raw.parquet"
        train_raw.to_parquet(train_raw_path, index=False)

        # Fit pipeline on new training data (same pattern as retrain_on_full_dataset)
        fitted_pipeline = fit_pipeline(
            dataset_uri=str(train_raw_path),
            pipeline=unfitted_pipeline,
            target_columns=[target_column],
        )
        logger.info("Pipeline fitted on new training data")

        # Transform both splits with fitted pipeline
        X_train_transformed = fitted_pipeline.transform(train_raw.drop(columns=[target_column]))
        X_val_transformed = fitted_pipeline.transform(val_raw.drop(columns=[target_column]))

        # Reconstruct DataFrames with generic column names
        num_features = X_train_transformed.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]

        train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_transformed[target_column] = train_raw[target_column].values

        val_transformed = pd.DataFrame(X_val_transformed, columns=feature_names)
        val_transformed[target_column] = val_raw[target_column].values

        # Save transformed data for training
        train_uri = retrain_dir / "train.parquet"
        val_uri = retrain_dir / "val.parquet"
        train_transformed.to_parquet(train_uri, index=False)
        val_transformed.to_parquet(val_uri, index=False)

        logger.info(f"Transformed data saved - Train: {train_transformed.shape}, Val: {val_transformed.shape}")

        # ============================================
        # Step 3: Train Model Using Existing Runner
        # ============================================
        logger.info("Training model on new data...")

        # Determine model artifact path based on model type
        artifacts_dir = extract_dir / "artifacts"
        model_artifact_names = {
            "xgboost": "model.pkl",
            "catboost": "model.cbm",
            "lightgbm": "model.pkl",
            "keras": "model.keras",
            "pytorch": "model_class.pkl",
        }
        artifact_name = model_artifact_names.get(model_type)
        if artifact_name is None:
            raise RetrainingError(f"Unsupported model type: {model_type}")

        original_model_path = artifacts_dir / artifact_name
        if not original_model_path.exists():
            raise RetrainingError(f"Original model not found: {original_model_path}")

        # Extract architecture and create NEW untrained model
        if model_type == "xgboost":
            # Load trained model to extract hyperparameters
            trained_model = joblib.load(original_model_path)
            params = trained_model.get_params()

            # Create fresh untrained model with same hyperparameters
            if isinstance(trained_model, XGBClassifier):
                untrained_model = XGBClassifier(**params)
            else:
                untrained_model = XGBRegressor(**params)
            logger.info("Created new untrained XGBoost model from architecture")

        elif model_type == "catboost":
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
            except ImportError:
                raise RetrainingError("CatBoost is required for retraining CatBoost models but is not installed")

            # CatBoost uses native format — try classifier first, then regressor
            try:
                trained_model = CatBoostClassifier()
                trained_model.load_model(str(original_model_path))
                params = trained_model.get_params()
                untrained_model = CatBoostClassifier(**params)
            except Exception as exc:
                logger.debug(
                    "CatBoost classifier load failed; falling back to regressor",
                    exc_info=exc,
                )
                trained_model = CatBoostRegressor()
                trained_model.load_model(str(original_model_path))
                params = trained_model.get_params()
                untrained_model = CatBoostRegressor(**params)
            logger.info("Created new untrained CatBoost model from architecture")

        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier, LGBMRanker
            except ImportError:
                raise RetrainingError("LightGBM is required for retraining LightGBM models but is not installed")

            trained_model = joblib.load(original_model_path)
            params = trained_model.get_params()

            if isinstance(trained_model, LGBMClassifier):
                untrained_model = LGBMClassifier(**params)
            elif isinstance(trained_model, LGBMRanker):
                untrained_model = LGBMRanker(**params)
            else:
                from lightgbm import LGBMRegressor

                untrained_model = LGBMRegressor(**params)
            logger.info("Created new untrained LightGBM model from architecture")

        elif model_type == "keras":
            # Load trained model to extract architecture config
            trained_model = keras.models.load_model(original_model_path)
            config = trained_model.get_config()

            # Rebuild model from config (NO weights loaded - fresh random initialization)
            untrained_model = keras.Model.from_config(config)
            logger.info("Created new untrained Keras model from architecture")
        elif model_type == "pytorch":
            import cloudpickle

            with open(original_model_path, "rb") as f:
                trained_model = cloudpickle.load(f)

            # Re-initialize weights (fresh random initialization, recursive for nested modules)
            def reset_weights(module):
                if hasattr(module, "reset_parameters"):
                    try:
                        module.reset_parameters()
                    except (AttributeError, NotImplementedError):
                        pass

            trained_model.apply(reset_weights)

            untrained_model = trained_model
            logger.info("Created new untrained PyTorch model from architecture")

        # Prepare training kwargs for neural networks
        training_kwargs = {}
        if model_type == "keras":
            # Load optimizer and loss from metadata (if available)
            if "optimizer_class" in metadata:
                # Reconstruct optimizer (simplified - may need more sophisticated logic)
                training_kwargs["optimizer"] = metadata.get("optimizer_class")
            if "loss_class" in metadata:
                training_kwargs["loss"] = metadata.get("loss_class")
            if "epochs" in metadata:
                training_kwargs["epochs"] = metadata["epochs"]
            if "batch_size" in metadata:
                training_kwargs["batch_size"] = metadata["batch_size"]
        elif model_type == "pytorch":
            import torch

            # Reconstruct actual optimizer and loss objects from saved metadata
            if "optimizer_class" in metadata:
                optimizer_cls = getattr(torch.optim, metadata["optimizer_class"])
                optimizer_config = metadata.get("optimizer_config", {})
                if optimizer_config:
                    signature = inspect.signature(optimizer_cls.__init__)
                    params = signature.parameters
                    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
                    if not accepts_kwargs:
                        allowed = {name for name in params if name not in ("self", "params")}
                        filtered_config = {key: value for key, value in optimizer_config.items() if key in allowed}
                        dropped_keys = sorted(set(optimizer_config) - set(filtered_config))
                        if dropped_keys:
                            logger.warning(
                                "Dropping unsupported optimizer args for %s: %s",
                                optimizer_cls.__name__,
                                dropped_keys,
                            )
                        optimizer_config = filtered_config
                training_kwargs["optimizer"] = optimizer_cls(untrained_model.parameters(), **optimizer_config)
            if "loss_class" in metadata:
                loss_cls = getattr(torch.nn, metadata["loss_class"])
                training_kwargs["loss"] = loss_cls()
            if "epochs" in metadata:
                training_kwargs["epochs"] = metadata["epochs"]
            if "batch_size" in metadata:
                training_kwargs["batch_size"] = metadata["batch_size"]

        # Run training using existing runner
        model_artifacts_path = runner.run_training(
            template=f"train_{model_type}",
            model=untrained_model,
            feature_pipeline=fitted_pipeline,
            train_uri=str(train_uri),
            val_uri=str(val_uri),
            timeout=config.training_timeout,
            target_columns=[target_column],
            **training_kwargs,
        )

        logger.info(f"Training complete! Artifacts at: {model_artifacts_path}")

        # ============================================
        # Step 4: Package Retrained Model
        # ============================================
        logger.info("Packaging retrained model...")

        package_dir = work_dir / "model"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy entire original model structure
        for item in extract_dir.iterdir():
            dest = package_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        logger.info("Copied original model structure")

        # Replace artifacts with retrained ones
        shutil.rmtree(package_dir / "artifacts")
        shutil.copytree(
            model_artifacts_path / "artifacts",
            package_dir / "artifacts",
        )
        logger.info("Replaced artifacts with retrained model")

        # Update model.yaml with retrain metadata
        model_yaml_path = package_dir / "model.yaml"
        if model_yaml_path.exists():
            with open(model_yaml_path) as f:
                model_metadata = yaml.safe_load(f)

            # Save original experiment ID
            original_experiment_id = model_metadata["metadata"].get("experiment_id")

            # Update metadata
            model_metadata["metadata"]["retrained"] = True
            model_metadata["metadata"]["retrained_at"] = pd.Timestamp.now().isoformat()
            model_metadata["metadata"]["original_experiment_id"] = original_experiment_id
            model_metadata["metadata"]["experiment_id"] = experiment_id

            with open(model_yaml_path, "w") as f:
                yaml.dump(model_metadata, f, default_flow_style=False, sort_keys=False)

            logger.info("Updated model.yaml with retrain metadata")

        logger.info(f"Retrained model package ready: {package_dir}")

        # ============================================
        # Step 5: Create Tarball Archive
        # ============================================
        logger.info("Creating tarball archive...")

        tarball_path = work_dir / "model.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add each item from package_dir at root level (not wrapped in model/)
            for item in package_dir.iterdir():
                tar.add(item, arcname=item.name)

        logger.info(f"Created tarball: {tarball_path}")
        logger.info(f"Tarball size: {tarball_path.stat().st_size / (1024**2):.2f} MB")

        # ============================================
        # Step 6: Create Solution and Metrics
        # ============================================
        # TODO: Evaluate retrained model on validation set (like workflow.py does)
        # Currently returning 0.0 performance - should use helpers.evaluate_on_sample()
        # to get actual validation performance on the retrained model

        # Load model.yaml to get metric name
        model_yaml_path = package_dir / "model.yaml"
        if model_yaml_path.exists():
            with open(model_yaml_path) as f:
                model_yaml = yaml.safe_load(f)
            metric_name = model_yaml["metric"]["name"]
        else:
            metric_name = "unknown"

        solution = Solution(
            iteration=0,
            feature_pipeline=fitted_pipeline,
            model=untrained_model,
            model_type=model_type,
            model_artifacts_path=model_artifacts_path,
            performance=0.0,  # TODO: Compute actual performance
        )

        final_metrics = {
            "metric": metric_name,
            "performance": 0.0,  # TODO: Compute actual performance
            "note": "Retrained model",
        }

        logger.info("Retraining complete!")
        return solution, final_metrics

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        raise RetrainingError(f"Retraining failed: {e}") from e


def _extract_package(local_model_path: str, extract_dir: Path) -> None:
    """
    Extract model package from tar.gz.

    Note: Expects local path. Adapter should handle S3 downloads before calling.

    Args:
        local_model_path: Local path to model.tar.gz (NOT S3 URI)
        extract_dir: Where to extract

    Raises:
        RetrainingError: If extraction fails
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Verify file exists
    if not Path(local_model_path).exists():
        raise RetrainingError(f"Model file not found: {local_model_path}")

    logger.info(f"Extracting {local_model_path}...")
    with tarfile.open(local_model_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    logger.info(f"Extracted to {extract_dir}")
