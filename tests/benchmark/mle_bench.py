#!/usr/bin/env python3
"""
This script automates the setup, execution, and grading process for the "mle-bench" framework using plexe.

Usage:
    python mle_bench.py --config CONFIG_PATH --rebuild

Description:
    The script clones and sets up "mle-bench", prepares datasets, reads a configuration file
    to determine the tests to run, executes models using plexe, and grades their performance. The
    --rebuild flag forces the script to re-clone the "mle-bench" repository and reinstall dependencies.

Ensure that your environment has the required permissions and Kaggle API credentials configured.
"""

import argparse
import sys

# Import the main runner class from the mlebench package
from mlebench.core.runner import MLEBenchRunner


def main(cli_args):
    """Main entry point for the script"""
    runner = MLEBenchRunner()
    runner.setup(cli_args)
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and grade an agent on the MLE-bench framework.")
    parser.add_argument(
        "--config", type=str, required=False, default="mle-bench-config.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force re-clone the MLE-bench repository and reinstall dependencies."
    )

    # Parse arguments and run main
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
