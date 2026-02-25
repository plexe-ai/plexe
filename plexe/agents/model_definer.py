"""
Model Definer Agent.

Defines model hyperparameters or architecture based on plan specification.
"""

import logging
from typing import Any

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.config import Config, ModelType
from plexe.constants import ScratchKeys
from plexe.models import BuildContext
from plexe.tools.submission import get_save_model_fn
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class ModelDefinerAgent:
    """
    Agent that defines model configuration.

    Implements ModelPlan specifications by creating untrained model objects.
    """

    def __init__(
        self,
        model_type: str,
        context: BuildContext,
        config: Config,
        transformed_schema: dict,
        plan: Any,  # ModelPlan specification
    ):
        """
        Initialize agent.

        Args:
            model_type: "xgboost" or "pytorch"
            context: Build context with task analysis
            config: Configuration
            transformed_schema: Schema of transformed data (columns, dtypes, num_features)
            plan: ModelPlan specification (directive, change_summary, rationale)
        """
        self.model_type = model_type
        self.context = context
        self.config = config
        self.llm_model = config.model_definer_llm
        self.transformed_schema = transformed_schema
        self.plan = plan

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with model definition submission function."""

        # Build stage-specific instructions
        instructions = self._build_instructions()

        # Add model-type specific imports
        extra_imports = []
        if self.model_type == ModelType.XGBOOST:
            extra_imports = ["xgboost"]
        elif self.model_type == ModelType.CATBOOST:
            extra_imports = ["catboost"]
        elif self.model_type == ModelType.LIGHTGBM:
            extra_imports = ["lightgbm"]
        elif self.model_type == ModelType.KERAS:
            extra_imports = [
                "keras",
                "keras.*",
                "keras.models",
                "keras.layers",
                "keras.optimizers",
                "keras.losses",
            ]

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="ModelDefiner",
            instructions=instructions,
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[],  # No tools - agent writes code directly
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports + extra_imports,
            max_steps=15,
        )

    @agent_span("ModelDefinerAgent")
    def run(self) -> tuple[Any, str]:
        """
        Create untrained model object from plan.

        Returns:
            (untrained_model, reasoning)
            - For XGBoost: XGBClassifier or XGBRegressor instance
            - For PyTorch: nn.Module instance
        """

        logger.info(f"Creating {self.model_type} model from plan...")

        # Build agent
        agent = self._build_agent()

        # Prepare arguments
        additional_args = {
            "save_model": get_save_model_fn(
                self.context,
                self.model_type,
                max_epochs=self.config.keras_default_epochs if self.model_type == ModelType.KERAS else None,
            ),
            "task_analysis": self.context.task_analysis,
        }

        # Run agent
        try:
            result = agent.run(task="Implement the model plan directive", additional_args=additional_args)

            # Retrieve saved model object
            model = self.context.scratch.get(ScratchKeys.SAVED_MODEL)

            if not model:
                raise ValueError("Agent did not save model")

            # Extract reasoning
            reasoning = str(result) if result else ""

            logger.info(f"{self.model_type} model created: {type(model).__name__}")
            return model, reasoning

        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise

    def _build_instructions(self) -> str:
        """Build instructions for implementing the model plan."""

        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Build transformed data schema section
        num_features = self.transformed_schema["num_features"]
        dtypes = self.transformed_schema["dtypes"]

        # Summarize dtypes (e.g., "45 float64, 2 int64")
        dtype_counts = {}
        for dtype in dtypes.values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dtype_summary = ", ".join(f"{count} {dtype}" for dtype, count in dtype_counts.items())

        task_type = self.context.task_analysis.get("task_type", "unknown")
        num_classes = self.context.task_analysis.get("num_classes", "unknown")

        # Determine model name for display
        if self.model_type == ModelType.XGBOOST:
            model_name = "XGBoost"
        elif self.model_type == ModelType.CATBOOST:
            model_name = "CatBoost"
        elif self.model_type == ModelType.LIGHTGBM:
            model_name = "LightGBM"
        elif self.model_type == ModelType.KERAS:
            model_name = "Keras"
        else:
            model_name = "PyTorch"

        instructions = (
            "## YOUR ROLE:\n"
            f"Create an untrained {model_name} model object.\n"
            "\n"
            f"{feedback_section}"
            "## PLAN TO IMPLEMENT:\n"
            f"Directive: {self.plan.directive}\n"
            f"Change Summary: {self.plan.change_summary}\n"
            f"Rationale: {self.plan.rationale}\n"
            "\n"
            "Implement this directive by creating the appropriate model configuration.\n"
            "\n"
            "## TASK CONTEXT:\n"
            f"Task Type: {task_type}\n"
            f"Number of Classes: {num_classes}\n"
            f"Transformed Features: {num_features} features ({dtype_summary})\n"
            "\n"
            "## INPUTS PROVIDED:\n"
            "- `task_analysis`: ML task analysis (task_type, num_classes)\n"
            "- `save_model(model)`: Submit your untrained model object\n"
            "\n"
        )

        # Model-type specific guidelines
        if self.model_type == ModelType.XGBOOST:
            instructions += (
                "## TASK:\n"
                "1. Import the appropriate XGBoost class based on task_analysis['task_type']\n"
                "   - Use XGBClassifier for classification tasks (binary_classification, multiclass_classification)\n"
                "   - Use XGBRegressor for regression tasks\n"
                "   - Use XGBRanker for ranking tasks (learning_to_rank)\n"
                "2. Interpret the plan directive and instantiate model with appropriate hyperparameters\n"
                "   - Common params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree\n"
                "   - For ranking: objective='rank:pairwise' or 'rank:ndcg'\n"
                "   - The directive is natural language - translate it to concrete parameter values\n"
                "3. Call save_model(model) with the untrained model instance\n"
                "\n"
                "Example directive interpretation:\n"
                "  'Increase n_estimators to around 250' → n_estimators=250\n"
                "  'Keep params similar to parent' → Use moderate defaults\n"
                "  'Try higher learning rate around 0.1' → learning_rate=0.1\n"
                "\n"
                "Example for RANKING tasks:\n"
                "  from xgboost import XGBRanker\n"
                "  model = XGBRanker(\n"
                "      objective='rank:pairwise',  # or 'rank:ndcg'\n"
                "      n_estimators=100,\n"
                "      max_depth=6,\n"
                "      learning_rate=0.1\n"
                "  )\n"
                "\n"
                "NOTE: XGBoost auto-detects feature dimensions - you don't need to specify input_dim.\n"
            )
        elif self.model_type == ModelType.CATBOOST:
            instructions += (
                "## TASK:\n"
                "1. Import CatBoostClassifier or CatBoostRegressor based on task_analysis['task_type']\n"
                "   - Use CatBoostClassifier for classification tasks\n"
                "   - Use CatBoostRegressor for regression tasks\n"
                "2. Interpret the plan directive and instantiate model with appropriate hyperparameters\n"
                "   - Common params: iterations, depth, learning_rate, l2_leaf_reg\n"
                "   - The directive is natural language - translate it to concrete parameter values\n"
                "3. Call save_model(model) with the untrained model instance\n"
                "\n"
                "Example directive interpretation:\n"
                "  'Increase iterations to around 500' → iterations=500\n"
                "  'Keep params similar to parent' → Use moderate defaults\n"
                "  'Try higher learning rate around 0.1' → learning_rate=0.1\n"
                "  'Use deeper trees around depth 8' → depth=8\n"
                "\n"
                "CATBOOST-SPECIFIC NOTES:\n"
                "- CatBoost auto-detects feature dimensions - you don't need to specify input_dim\n"
                "- CatBoost handles categorical features natively (no encoding needed)\n"
                "- Param name is 'iterations' (not 'n_estimators' like XGBoost)\n"
                "- Param name is 'depth' (not 'max_depth' like XGBoost)\n"
                "- Default depth=6, learning_rate=0.03, iterations=1000\n"
            )
        elif self.model_type == ModelType.LIGHTGBM:
            instructions += (
                "## TASK:\n"
                "1. Import the appropriate LightGBM class based on task_analysis['task_type']\n"
                "   - Use LGBMClassifier for classification tasks (binary_classification, multiclass_classification)\n"
                "   - Use LGBMRegressor for regression tasks\n"
                "   - Use LGBMRanker for ranking tasks (learning_to_rank)\n"
                "2. Interpret the plan directive and instantiate model with appropriate hyperparameters\n"
                "   - Common params: n_estimators, num_leaves, max_depth, learning_rate, subsample, colsample_bytree\n"
                "   - For ranking: objective='lambdarank' or 'rank_xendcg'\n"
                "   - The directive is natural language - translate it to concrete parameter values\n"
                "3. Call save_model(model) with the untrained model instance\n"
                "\n"
                "Example directive interpretation:\n"
                "  'Increase n_estimators to around 200' → n_estimators=200\n"
                "  'Try more leaves around 63' → num_leaves=63\n"
                "  'Try higher learning rate around 0.1' → learning_rate=0.1\n"
                "\n"
                "LIGHTGBM-SPECIFIC NOTES:\n"
                "- LightGBM auto-detects feature dimensions - you don't need to specify input_dim\n"
                "- Key param is 'num_leaves' (controls tree complexity, default 31)\n"
                "- num_leaves should be <= 2^max_depth to avoid overfitting\n"
                "- Default n_estimators=100, learning_rate=0.1, num_leaves=31\n"
                "- Set verbose=-1 to suppress training output\n"
            )
        elif self.model_type == ModelType.KERAS:
            instructions += (
                "## TASK:\n"
                "Create THREE objects: model, optimizer, and loss.\n"
                "\n"
                "1. **Import Keras 3** (CRITICAL - use submodule imports, set backend first):\n"
                "   import os\n"
                "   os.environ['KERAS_BACKEND'] = 'tensorflow'  # MUST be tensorflow\n"
                "   from keras.models import Sequential\n"
                "   from keras.layers import Dense, Dropout, BatchNormalization, Input\n"
                "   from keras.optimizers import Adam, SGD, RMSprop\n"
                "   from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError\n"
                "\n"
                "2. **Create Model** (Sequential with list of layers):\n"
                f"   - Input shape: ({num_features},)\n"
                f"   - Output units: {num_classes if 'classification' in task_type else 1}\n"
                "   - Interpret the plan directive for architecture (num layers, units, activation, dropout)\n"
                "\n"
                "   Example:\n"
                "   model = Sequential([\n"
                f"       Input(shape=({num_features},)),\n"
                "       Dense(64, activation='relu'),\n"
                "       Dropout(0.2),\n"
                "       Dense(32, activation='relu'),\n"
                f"       Dense({num_classes})\n"
                "   ])\n"
                "\n"
                "3. **Create Optimizer**:\n"
                "   optimizer = Adam(learning_rate=0.001)  # or SGD, RMSprop, etc.\n"
                "\n"
                "4. **Create Loss**:\n"
                "   - Binary classification: loss = BinaryCrossentropy()\n"
                "   - Multi-class: loss = SparseCategoricalCrossentropy(from_logits=True)\n"
                "   - Regression: loss = MeanSquaredError()\n"
                "\n"
                "5. **Determine Training Config**:\n"
                "   - Extract epochs and batch_size from the plan directive (e.g., 'Train 40 epochs, batch_size 64')\n"
                f"   - If not specified, use defaults: epochs={self.config.keras_default_epochs}, batch_size={self.config.keras_default_batch_size}\n"
                f"   - epochs MUST be ≤ {self.config.keras_default_epochs} (enforced cap)\n"
                "   - Consider dataset size and model complexity when choosing\n"
                "\n"
                "6. Call save_model(model, optimizer, loss, epochs, batch_size) with all five arguments\n"
                "\n"
                "CRITICAL:\n"
                "- Use submodule imports (from keras.X import Y), NOT 'import keras'\n"
                "- KERAS_BACKEND MUST be 'tensorflow'\n"
                "- Create actual object instances, not strings\n"
                "- epochs and batch_size must be integers\n"
                f"- epochs MUST NOT exceed {self.config.keras_default_epochs}\n"
            )
        else:  # PyTorch
            instructions += (
                "## TASK:\n"
                "1. Import torch and torch.nn\n"
                f"2. Create neural network with input_dim={num_features}\n"
                f"3. Set output_dim based on task:\n"
                f"   - Classification: output_dim={num_classes}\n"
                "   - Regression: output_dim=1\n"
                "4. Interpret the plan directive to determine architecture (layers, activation, etc.)\n"
                "5. Call save_model(model) with the untrained nn.Module instance\n"
                "\n"
                "The directive is natural language - translate it to concrete architecture.\n"
            )

        return instructions
