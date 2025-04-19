"""
This script demonstrates how to run the plexe ML engineering agent to build a predictive model. The example
uses the Kaggle 'Santander Customer Transaction Prediction' competition's training dataset.

The dataset is owned by Banco Santander and is hosted by Kaggle. The dataset is available for download at
https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview and is licenced subject to the
competition rules (https://www.kaggle.com/competitions/santander-customer-transaction-prediction/rules#7-competition-data).
This dataset is not part of the plexe package or in any way affiliated to it, and Plexe AI claims no rights over it.
The dataset is used here for demonstration purposes only. Please refer to the Kaggle competition page for more details
on the dataset and its usage.

Citation:
Mercedes Piedra, Sohier Dane, and Soraya_Jimenez. Santander Customer Transaction Prediction.
https://kaggle.com/competitions/santander-customer-transaction-prediction, 2019. Kaggle.
"""

from datetime import datetime

import pandas as pd

import plexe
from plexe.internal.common.provider import ProviderConfig

# Step 1: Define the model
# NOTE: this dataset has over 200 columns, so we leave input schema empty and let plexe infer it automatically
model = plexe.Model(
    intent=(
        "Identify which customers will make a specific transaction in the future, irrespective of the amount "
        "of money transacted. For each Id, make a binary prediction of the 'target' variable."
    ),
    output_schema={
        "target": int,
    },
)

# Step 2: Build the model using the training dataset
# 2A [OPTIONAL]: Define MLFlow callback for tracking
mlflow_callback = plexe.callbacks.MLFlowCallback(
    tracking_uri="http://127.0.0.1:8080",
    experiment_name=f"santander-transactions-{datetime.now().strftime('%Y%m%d-%H%M%S')  }",
)
# 2B: Build the model with the dataset
# NOTE: In order to run this example, you will need to download the dataset from Kaggle
model.build(
    datasets=[pd.read_csv("examples/datasets/santander-transactions-train-mini.csv")],
    provider=ProviderConfig(
        default_provider="openai/gpt-4o",
        orchestrator_provider="anthropic/claude-3-7-sonnet-20250219",
        research_provider="openai/gpt-4o",
        engineer_provider="anthropic/claude-3-7-sonnet-20250219",
        ops_provider="anthropic/claude-3-7-sonnet-20250219",
        tool_provider="openai/gpt-4o",
    ),
    max_iterations=8,
    timeout=1800,  # 30 minute timeout
    run_timeout=180,
    verbose=False,
    callbacks=[mlflow_callback],
)

# Step 3: Save the model
plexe.save_model(model, "santander_transactions_model.tar.gz")

# Step 4: Run a prediction on the built model
test_df = pd.read_csv("examples/datasets/santander-transactions-test-mini.csv")
predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])

# Step 5: print a sample of predictions
print(predictions.sample(10))

# Step 6: Print model description
description = model.describe()
print(description.as_text())
