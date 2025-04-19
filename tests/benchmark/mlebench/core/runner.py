"""Main runner class for MLE Bench benchmark."""

import os
import sys
from pathlib import Path

from mlebench.core.config import ConfigManager
from mlebench.core.validator import EnvironmentValidator
from mlebench.runners.setup import MLEBenchSetup
from mlebench.runners.test_runner import TestRunner
from mlebench.runners.grader import MLEBenchGrader


class MLEBenchRunner:
    """Main class to run the MLE-bench benchmarking framework"""

    def __init__(self):
        self.config = None
        self.workdir = None

    def setup(self, cli_args):
        """Set up the MLE-bench environment"""
        print("🚀 Starting the MLE-bench Runner with Plexe...")

        # Get the absolute path for the config file
        config_path = Path(cli_args.config).absolute()
        print(f"📄 Using configuration file: {config_path}")

        # Create workdir if it doesn't exist
        self.workdir = Path(os.getcwd()) / "workdir"
        self.workdir.mkdir(exist_ok=True)
        print(f"📁 Using working directory: {self.workdir}")

        # Check if LLM API key is set
        if not EnvironmentValidator.check_llm_api_keys():
            print("❌ Required LLM API key environment variables not set. Please set them before running.")
            print("❌ See the README.md file for instructions on configuring API keys.")
            sys.exit(1)

        # Ensure that the configuration file exists, then load it
        ConfigManager.ensure_config_exists(cli_args.rebuild)
        self.config = ConfigManager.load_config(config_path)

        # Ensure Kaggle credentials are set up
        EnvironmentValidator.ensure_kaggle_credentials()

        # Set up MLE-bench and prepare datasets
        MLEBenchSetup.setup_mle_bench(self.config, cli_args.rebuild)
        MLEBenchSetup.prepare_datasets(self.config)

    def run(self):
        """Run the tests and evaluate the results"""
        # Run tests
        test_runner = TestRunner(self.config)
        submissions = test_runner.run_tests()

        # Grade agent if there are submissions
        if submissions:
            grades_dir = MLEBenchGrader.grade_agent(submissions)
            print(f"📊 Benchmark results saved to: {grades_dir}")
        else:
            print("❌ No submissions were generated. Cannot grade the agent.")

        print("✅ Script completed. Thank you for using the MLE-bench Runner!")
