import pandas as pd

import plexe
from plexe.internal.common.provider import ProviderConfig
from plexe.callbacks import MLFlowCallback

# Step 1: Load an existing model for Spaceship Titanic
model = plexe.load_model("examples/artifacts/st_model.tar.gz")

# Step 2: Continue building the model from where it left off
# NOTE: In order to run this example, you will need to download the dataset from Kaggle
model.build(
    resume=True,
    datasets=[pd.read_csv("examples/datasets/spaceship-titanic-train.csv")],
    provider=ProviderConfig(
        default_provider="openai/gpt-4o",
        orchestrator_provider="anthropic/claude-3-7-sonnet-20250219",
        research_provider="openai/gpt-4o",
        engineer_provider="anthropic/claude-3-7-sonnet-20250219",
        ops_provider="anthropic/claude-3-7-sonnet-20250219",
        tool_provider="openai/gpt-4o",
    ),
    max_iterations=1,
    callbacks=[
        MLFlowCallback(
            tracking_uri="http://127.0.0.1:8080",
            experiment_name="spaceship-titanic-example",
        )
    ],
    verbose=True,
)

# Step 3: Save the model
plexe.save_model(model, "st_model_continued.tar.gz")

# Step 4: Print model description
description = model.describe()
print(description.as_text())
