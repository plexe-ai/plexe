"""
Spaceship Titanic example — build an ML model from a natural language description.

Dataset: https://www.kaggle.com/c/spaceship-titanic
License: CC BY 4.0
"""

import sys
import time
from pathlib import Path

from plexe.main import main

if __name__ == "__main__":
    dataset_uri = "examples/datasets/spaceship-titanic/train.parquet"
    work_dir = Path(f"./workdir/spaceship_titanic/{time.time()}")

    try:
        best_solution, final_metrics, _ = main(
            intent="predict whether a passenger was transported or not based on other features",
            data_refs=[dataset_uri],
            user_id="example_user",
            experiment_id="spaceship_titanic",
            max_iterations=6,
            work_dir=work_dir,
            enable_final_evaluation=True,
            allowed_model_types=["xgboost"],
        )

        print(f"\nModel built successfully | Performance: {best_solution.performance:.4f}")
        print(f"Model artifacts: {work_dir / 'model'}")

    except Exception as e:
        print(f"\nModel building failed: {e}")
        sys.exit(1)
