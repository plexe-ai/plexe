"""
Local process runner - executes training in subprocess.
"""

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from plexe.execution.training.runner import TrainingRunner
from plexe.models import TrainingError

logger = logging.getLogger(__name__)


class LocalProcessRunner(TrainingRunner):
    """
    Runs training in local subprocess.

    Suitable for:
    - Development/testing
    - Small datasets that fit on single machine
    - When external compute (SageMaker, EMR) is not needed
    """

    def __init__(self, work_dir: str = "/tmp/model_training"):
        """
        Initialize local runner.

        Args:
            work_dir: Base directory for training runs
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_training(
        self,
        template: str,
        model: Any,
        feature_pipeline: Pipeline,
        train_uri: str,
        val_uri: str,
        timeout: int,
        target_columns: list[str],
        optimizer: Any = None,
        loss: Any = None,
        epochs: int = None,
        batch_size: int = None,
        group_column: str | None = None,
    ) -> Path:
        """
        Execute training in subprocess.

        Args:
            template: "train_xgboost", "train_catboost", "train_lightgbm", or "train_pytorch"
            model: Untrained model object (XGBClassifier, LGBMClassifier, or nn.Module)
            feature_pipeline: Fitted sklearn Pipeline
            train_uri: Training data URI
            val_uri: Validation data URI
            timeout: Max training time (seconds)
            target_columns: List of target column names

        Returns:
            Path to model artifacts directory

        Raises:
            TrainingError: If training fails
        """

        # ============================================
        # Step 1: Create Run-Specific Work Directory
        # ============================================
        run_id = str(uuid.uuid4())[:8]
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training run {run_id} started in {run_dir}")

        try:
            # ============================================
            # Step 2: Save Untrained Model to Disk
            # ============================================
            untrained_model_path = run_dir / "untrained_model.pkl"

            # Save based on model type
            if "xgboost" in template:
                joblib.dump(model, untrained_model_path)
                logger.info(f"Saved untrained XGBoost model to {untrained_model_path}")
            elif "catboost" in template:
                # IMPORTANT: CatBoost doesn't allow .save_model() on untrained models
                # Use pickle for untrained (safe - no best_iteration_ attribute yet)
                # Training script will use native .save_model() for trained model
                joblib.dump(model, untrained_model_path)
                logger.info(f"Saved untrained CatBoost model to {untrained_model_path}")
            elif "lightgbm" in template:
                joblib.dump(model, untrained_model_path)
                logger.info(f"Saved untrained LightGBM model to {untrained_model_path}")
            elif "keras" in template:
                # For Keras, use native Keras serialization (cloudpickle causes deadlocks)
                if not optimizer or not loss:
                    raise TrainingError("Keras training requires optimizer and loss to be passed as parameters")

                # Save model using Keras native format
                model_path = run_dir / "untrained_model.keras"
                model.save(model_path)

                # Save optimizer and loss configs as JSON
                training_config = {
                    "optimizer_class": type(optimizer).__name__,
                    "optimizer_config": optimizer.get_config(),
                    "loss_class": type(loss).__name__,
                    "loss_config": loss.get_config(),
                }

                config_path = run_dir / "training_config.json"
                with open(config_path, "w") as f:
                    json.dump(training_config, f, indent=2)

                logger.info(f"Saved untrained Keras model to {model_path}")
                logger.info(f"Saved training config to {config_path}")

                # Update untrained_model_path to point to the keras file
                untrained_model_path = model_path
            elif "pytorch" in template:
                import torch

                torch.save(model, untrained_model_path)
                logger.info(f"Saved untrained PyTorch model to {untrained_model_path}")
            else:
                raise TrainingError(f"Unknown template type: {template}")

            output_dir = run_dir / "model_artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)

            # ============================================
            # Step 3: Build Command
            # ============================================
            # Find template script
            template_script = Path(__file__).parent.parent.parent / "templates" / "training" / f"{template}.py"

            if not template_script.exists():
                raise TrainingError(f"Training template not found: {template_script}")

            # Note: Currently only single-target supported
            target_column = target_columns[0] if target_columns else "target"

            cmd = [
                "python",
                str(template_script),
                "--untrained-model",
                str(untrained_model_path),
                "--train-uri",
                train_uri,
                "--val-uri",
                val_uri,
                "--target-column",
                target_column,
                "--output",
                str(output_dir),
            ]

            # Add Keras-specific training params if provided
            if "keras" in template:
                if epochs is not None:
                    cmd.extend(["--epochs", str(epochs)])
                if batch_size is not None:
                    cmd.extend(["--batch-size", str(batch_size)])

            # Add ranking-specific params if provided
            if group_column is not None:
                cmd.extend(["--group-column", group_column])

            logger.debug(f"Executing: {' '.join(cmd)}")
            logger.info(f"Starting training subprocess (timeout: {timeout}s)...")

            # ============================================
            # Step 4: Execute Training (with real-time output streaming)
            # ============================================
            # Set PYTHONUNBUFFERED to ensure real-time streaming (especially for Keras progress bars)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified streaming
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )

            # Stream output in real-time while capturing it
            stdout_lines = []
            start_time = time.time()

            try:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        # Print to terminal in real-time (shows Keras progress bars)
                        print(line.rstrip())
                        # Capture for error logging
                        stdout_lines.append(line)

                    # Check timeout manually
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        process.kill()
                        process.wait()
                        raise subprocess.TimeoutExpired(cmd, timeout)

                # Wait for process to complete
                return_code = process.wait(timeout=5)  # Short wait since process already finished

            finally:
                if process.stdout:
                    process.stdout.close()

            # ============================================
            # Step 5: Check Result
            # ============================================
            stdout_captured = "".join(stdout_lines)

            if return_code != 0:
                error_msg = f"Training failed with return code {return_code}\n"
                error_msg += f"OUTPUT:\n{stdout_captured}"
                logger.error(error_msg)
                raise TrainingError(error_msg)

            logger.info(f"Training completed successfully for run {run_id}")

            # ============================================
            # Step 5: Verify Outputs Exist (in artifacts/ subdirectory)
            # ============================================
            artifacts_dir = output_dir / "artifacts"
            required_files = ["metadata.json"]

            # Model file depends on template
            if "xgboost" in template:
                required_files.append("model.pkl")
            elif "catboost" in template:
                required_files.append("model.cbm")
            elif "lightgbm" in template:
                required_files.append("model.pkl")
            elif "keras" in template:
                required_files.append("model.keras")
            elif "pytorch" in template:
                required_files.append("model.pt")

            missing_files = [f for f in required_files if not (artifacts_dir / f).exists()]

            if missing_files:
                raise TrainingError(f"Training did not produce required files in artifacts/: {missing_files}")

            # ============================================
            # Step 7: Return Artifacts Path
            # ============================================
            return output_dir

        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out after {timeout} seconds")
            raise TrainingError(f"Training timed out after {timeout} seconds")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training failed: {e}") from e

        finally:
            # Note: We keep run_dir for debugging
            # Production deployments may want to clean up
            pass
