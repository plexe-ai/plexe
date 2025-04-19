"""Test runner for MLE Bench benchmarks."""

import os
import time
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import platformdirs
import plexe
from tqdm import tqdm

from mlebench.core.models import TestResult, SubmissionInfo
from mlebench.utils.error import ErrorHandler

# Sklearn often throws warnings when used in plexe
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class TestRunner:
    """Class to run tests using plexe"""

    def __init__(self, config):
        self.config = config
        self.provider = config.get("provider", "openai/gpt-4o")
        self.max_iterations = config.get("max_iterations", 3)
        self.timeout = config.get("timeout", 3600)  # Default 1 hour timeout
        self.workdir = Path(os.getcwd()) / "workdir"
        self.workdir.mkdir(exist_ok=True)
        self.mle_bench_data_dir = Path(platformdirs.user_cache_dir()) / "mle-bench" / "data"

        print(f"🔧 Using provider: {self.provider}, max_iterations: {self.max_iterations}, timeout: {self.timeout}s")

    def verify_test_files(self, test_name) -> bool:
        """Verify that all required files for a test exist"""
        data_dir = self.mle_bench_data_dir / test_name / "prepared" / "public"
        required_files = [
            data_dir / "train.csv",
            data_dir / "test.csv",
            data_dir / "description.md",
            data_dir / "sample_submission.csv",
        ]

        for file_path in required_files:
            if not file_path.exists():
                print(f"❌ Required file not found: {file_path}")
                return False

        return True

    def prepare_test(self, test_name):
        """Prepare test data and create output directory"""
        data_dir = self.mle_bench_data_dir / test_name / "prepared" / "public"

        # Read task description
        with open(data_dir / "description.md", "r") as f:
            task_description = f.read()

        # Create output directory for this test
        output_dir = self.workdir / test_name
        output_dir.mkdir(exist_ok=True)

        # Load datasets
        print(f"📊 Loading datasets for {test_name}...")
        train_data = pd.read_csv(data_dir / "train.csv")
        test_data = pd.read_csv(data_dir / "test.csv")
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

        test_data_info = {
            "train_data": train_data,
            "test_data": test_data,
            "sample_submission": sample_submission,
            "task_description": task_description,
            "output_dir": output_dir,
        }

        return test_data_info

    def build_model(self, test_name, test_data_info):
        """Build a model using plexe"""
        print(f"🤖 Creating model for {test_name}...")
        model = plexe.Model(
            intent=test_data_info["task_description"],
        )

        # Build the model
        print(f"🏗️ Building model for {test_name}...")
        start_time = time.time()
        try:
            model.build(
                datasets=[test_data_info["train_data"]],
                provider=self.provider,
                max_iterations=self.max_iterations,
                timeout=self.timeout,
            )
            build_time = time.time() - start_time
            print(f"✅ Model built successfully in {build_time:.2f} seconds")
            return model
        except Exception as e:
            ErrorHandler.handle_error("model building", test_name, e)
            return None

    def validate_predictions(self, predictions, expected_columns):
        """Validate prediction data has required columns and format"""
        missing_cols = [col for col in expected_columns if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Predictions missing required columns: {missing_cols}")
        return True

    def generate_predictions(self, model, test_name, test_data_info):
        """Generate predictions using the model"""
        print(f"🔮 Generating predictions for {test_name}...")

        test_data = test_data_info["test_data"]
        sample_submission = test_data_info["sample_submission"]
        output_dir = test_data_info["output_dir"]

        # Determine columns that need to be in submission file from the sample submission
        submission_columns = list(sample_submission.columns)

        print(f"🎯 Target columns: {submission_columns}")

        # Create submission file path
        submission_path = output_dir / "submission.csv"

        try:
            # Process each row in test data
            prediction_results = []

            print(f"📊 Processing {len(test_data)} test records...")

            # Use tqdm for progress tracking
            for idx, (_, row) in enumerate(
                tqdm(test_data.iterrows(), total=len(test_data), desc=f"Generating predictions for {test_name}")
            ):
                try:
                    # Convert row to dictionary
                    row_dict = row.to_dict()

                    # Make prediction for this row
                    row_prediction = model.predict(row_dict)

                    # Concatenate row_dict with row_prediction, then keep only submissions columns
                    row_prediction = {**row_dict, **row_prediction}
                    row_prediction = {k: v for k, v in row_prediction.items() if k in submission_columns}

                    prediction_results.append(row_prediction)

                except Exception as e:
                    print(f"⚠️ Error predicting row {idx}: {e}")
                    # Add empty prediction to maintain row count
                    empty_prediction = {col: None for col in submission_columns}
                    prediction_results.append(empty_prediction)

            # Create a DataFrame from all the prediction results
            all_predictions_df = pd.DataFrame(prediction_results)

            # Validate predictions have required columns
            self.validate_predictions(all_predictions_df, sample_submission.columns)

            # Save the prediction results
            all_predictions_df.to_csv(submission_path, index=False)
            print(f"✅ Predictions generated and submission file created at {submission_path}")
            return submission_path

        except Exception as e:
            ErrorHandler.handle_error("prediction generation", test_name, e)
            return None

    def save_model(self, model, test_name, output_dir):
        """Save model for future reference"""
        model_save_path = output_dir / f"{test_name}_model.tar.gz"
        try:
            plexe.save_model(model, str(model_save_path))
            print(f"✅ Model saved to {model_save_path}")
            return model_save_path
        except Exception as e:
            print(f"⚠️ Failed to save model (non-critical): {e}")
            return None

    def run_tests(self) -> List[SubmissionInfo]:
        """Run tests from the configuration file using plexe"""
        print("🏁 Starting test execution with plexe...")
        test_results = []
        submissions = []

        for test_name in self.config.get("datasets", []):
            try:
                print(f"🔍 Running test: {test_name}")

                # Check if required files exist
                if not self.verify_test_files(test_name):
                    test_results.append(TestResult(name=test_name, success=False, failure_reason="missing files"))
                    continue

                # Prepare test data
                test_data_info = self.prepare_test(test_name)

                # Build model
                model = self.build_model(test_name, test_data_info)

                if model:
                    # Generate predictions
                    submission_path = self.generate_predictions(model, test_name, test_data_info)

                    if submission_path:
                        # Save model
                        model_path = self.save_model(model, test_name, test_data_info["output_dir"])

                        # Record successful test
                        test_results.append(
                            TestResult(
                                name=test_name,
                                success=True,
                                submission_path=str(submission_path),
                                model_path=str(model_path) if model_path else None,
                            )
                        )

                        # Add to submissions for grading
                        submissions.append(
                            SubmissionInfo(competition_id=test_name, submission_path=str(submission_path))
                        )
                    else:
                        test_results.append(
                            TestResult(name=test_name, success=False, failure_reason="prediction failed")
                        )
                else:
                    test_results.append(TestResult(name=test_name, success=False, failure_reason="model build failed"))

            except Exception as e:
                ErrorHandler.handle_error("test execution", test_name, e)
                test_results.append(
                    TestResult(name=test_name, success=False, failure_reason=f"general error: {str(e)}")
                )

        # Report failed tests
        failed_tests = [test for test in test_results if not test.success]
        if failed_tests:
            print(f"⚠️ {len(failed_tests)} tests failed:")
            for test in failed_tests:
                print(f"⚠️   - {test.name}: {test.failure_reason}")

        return submissions
