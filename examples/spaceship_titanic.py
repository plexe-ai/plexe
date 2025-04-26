"""
This script demonstrates how to run the plexe ML engineering agent to build a predictive model. The example
uses the Kaggle 'Spaceship Titanic' competition's training dataset.

The dataset is owned and hosted by Kaggle, and is available at https://www.kaggle.com/c/spaceship-titanic/overview
under the Attribution 4.0 International (CC BY 4.0) license (https://creativecommons.org/licenses/by/4.0/). This
dataset is not part of the plexe package or in any way affiliated to it, and Plexe AI claims no rights over it.
The dataset is used here for demonstration purposes only. Please refer to the Kaggle competition page for more details
on the dataset and its usage.

Citation:
Addison Howard, Ashley Chow, and Ryan Holbrook. Spaceship Titanic.
https://kaggle.com/competitions/spaceship-titanic, 2022. Kaggle.
"""

from datetime import datetime

import pandas as pd

import plexe
from plexe.internal.common.provider import ProviderConfig

# Step 1: Define the model using the Spaceship Titanic problem statement as the model description
model = plexe.Model(
    intent=(
        "From features describing a Spaceship Titanic passenger's information, determine whether they were "
        "transported or not."
    ),
    input_schema={
        "PassengerId": str,
        "HomePlanet": str,
        "CryoSleep": bool,
        "Cabin": str,
        "Destination": str,
        "Age": float,
        "VIP": bool,
        "RoomService": float,
        "FoodCourt": float,
        "ShoppingMall": float,
        "Spa": float,
        "VRDeck": float,
        "Name": str,
    },
    output_schema={
        "Transported": bool,
    },
)

# Step 2: Build the model using the Spaceship Titanic training dataset
# 2A [OPTIONAL]: Define MLFlow callback for tracking
mlflow_callback = plexe.callbacks.MLFlowCallback(
    tracking_uri="http://127.0.0.1:8080",
    experiment_name=f"spaceship-titanic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)
# 2B: Build the model with the dataset
# NOTE: In order to run this example, you will need to download the dataset from Kaggle
model.build(
    datasets=[pd.read_csv("examples/datasets/spaceship-titanic-train.csv")],
    provider=ProviderConfig(
        default_provider="openai/gpt-4o",
        orchestrator_provider="anthropic/claude-3-7-sonnet-20250219",
        research_provider="openai/gpt-4o",
        engineer_provider="anthropic/claude-3-7-sonnet-20250219",
        ops_provider="anthropic/claude-3-7-sonnet-20250219",
        tool_provider="openai/gpt-4o",
    ),
    max_iterations=4,
    timeout=300,  # 5 minute timeout
    run_timeout=150,
    verbose=False,
    callbacks=[mlflow_callback],
    chain_of_thought=True,
)

# Step 3: Save the model
plexe.save_model(model, "spaceship_titanic_model.tar.gz")

# Step 4: Run a prediction on the built model
test_df = pd.read_csv("examples/datasets/spaceship-titanic-test.csv")
predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])

# Step 5: print a sample of predictions
print(predictions.sample(10))

# Step 6: Print model description
description = model.describe()
print(description.as_text())
