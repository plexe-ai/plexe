"""
This script demonstrates how to run the plexe ML engineering agent to build a predictive model. The example
uses the Kaggle 'House Prices - Advanced Regression Techniques' competition's training dataset.

The dataset is owned and hosted by Kaggle, and is available for download at
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data under the MIT license
(https://www.mit.edu/~amini/LICENSE.md). This dataset is not part of the plexe package or in any way
affiliated to it, and Plexe AI claims no rights over it. The dataset is used here for demonstration purposes
only. Please refer to the Kaggle competition page for more details on the dataset and its usage.

Citation:
Anna Montoya and DataCanary. House Prices - Advanced Regression Techniques.
https://kaggle.com/competitions/house-prices-advanced-regression-techniques, 2016. Kaggle.
"""

# NOTE: you must download the dataset from Kaggle for this example to work

from datetime import datetime
import pandas as pd

import plexe
from plexe.internal.common.provider import ProviderConfig


# Step 1: Define the model
# Note: for conciseness we leave the input schema empty and let plexe infer it
model = plexe.Model(
    intent=(
        "With 79 explanatory variables describing aspects of residential homes in Ames, Iowa, predict "
        "the final price of each home. Use only linear regression and decision tree models, no ensembling. "
        "The models must be extremely simple and quickly trainable on extremely constrained hardware."
    ),
    output_schema={
        "SalePrice": float,
    },
)

# Step 2: Build the model using the training dataset
# 2A [OPTIONAL]: Define MLFlow callback for tracking
mlflow_callback = plexe.callbacks.MLFlowCallback(
    tracking_uri="http://127.0.0.1:8080",
    experiment_name=f"house-prices-{datetime.now().strftime('%Y%m%d-%H%M%S')  }",
)
# 2B: Build the model with the dataset
# NOTE: In order to run this example, you will need to download the dataset from Kaggle
model.build(
    datasets=[pd.read_csv("examples/datasets/house-prices-train.csv")],
    provider=ProviderConfig(
        default_provider="openai/gpt-4o",
        orchestrator_provider="anthropic/claude-3-7-sonnet-20250219",
        research_provider="openai/gpt-4o",
        engineer_provider="anthropic/claude-3-7-sonnet-20250219",
        ops_provider="anthropic/claude-3-7-sonnet-20250219",
        tool_provider="openai/gpt-4o",
    ),
    max_iterations=2,
    timeout=1800,  # 30 minute timeout
    run_timeout=180,
    verbose=False,
    callbacks=[mlflow_callback],
    chain_of_thought=True,  # Enable chain of thought output
)

# Step 3: Save the model
plexe.save_model(model, "house-prices.tar.gz")

# Step 4: Run a prediction on the built model
test_df = pd.read_csv("examples/datasets/house-prices-test.csv").sample(10)
predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])

# Step 5: print a sample of predictions
print(predictions)

# Step 6: Print model description
description = model.describe()
print(description.as_text())
