"""
Baseline Builder Agent.

Builds simple heuristic baseline predictors (code-based, no pickling).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.config import Config, get_routing_for_model
from plexe.constants import DirNames
from plexe.models import BuildContext, Baseline
from plexe.utils.tracing import agent_span
from plexe.tools.submission import (
    get_validate_baseline_predictor_tool,
    get_save_baseline_code_tool,
)

logger = logging.getLogger(__name__)


class BaselineBuilderAgent:
    """
    Agent that creates simple baseline predictors.

    Works with pandas DataFrames on samples - no distributed computing.
    """

    def __init__(self, spark: SparkSession, context: BuildContext, config: Config):
        """
        Initialize agent.

        Args:
            spark: SparkSession for loading sample data
            context: Build context
            config: Configuration
        """
        self.spark = spark
        self.context = context
        self.config = config
        self.llm_model = config.baseline_builder_llm

    def _build_agent(self, val_sample_df) -> CodeAgent:
        """Build CodeAgent with baseline tools."""
        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        # Check for previous error history from retries
        error_history = self.context.scratch.get("_baseline_errors", [])
        error_context = ""
        if error_history:
            error_context = (
                "\n## ⚠️ PREVIOUS ATTEMPT FAILURES:\n"
                "Previous attempts to generate the baseline failed with these errors:\n"
            )
            for err in error_history[-3:]:  # Show last 3 errors
                error_context += f"- Attempt {err['attempt']}: {err['error_type']}: {err['error']}\n"
            error_context += "\n**IMPORTANT:** Avoid making the same mistakes. "

        return CodeAgent(
            name="BaselineBuilder",
            instructions=(
                error_context + "\n" + feedback_section + "## YOUR ROLE:\n"
                "Create a simple HEURISTIC baseline predictor (no learned models) that establishes a performance floor.\n"
                "\n"
                "## PHILOSOPHY - HEURISTICS ONLY:\n"
                "1. Use ONLY pure heuristics (no sklearn models - they cause serialization issues)\n"
                "2. Use data insights AND the ML task description to choose a meaningful heuristic\n"
                "3. Create a simple Python class that uses coded logic to produce predictions\n"
                "4. Focus on interpretable, domain-appropriate heuristics\n"
                "5. Examples: most frequent class, mean value, classify on most predictive feature, etc\n"
                "\n"
                "## EXAMPLES:\n"
                "- Churn prediction → predict 'no churn' (majority class)\n"
                "- Sales forecasting → predict historical average\n"
                "- Fraud detection → predict 'not fraud' (majority class)\n"
                "\n"
                "## TEMPLATE:\n"
                "Use this template structure for your predictor:\n"
                "```python\n"
                "# TODO: add any additional required imports here\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "class HeuristicBaselinePredictor:\n"
                "    def predict(self, x: pd.DataFrame) -> np.ndarray:\n"
                '        """\n'
                "        Return a heuristic-based prediction given the input data.\n"
                '        """\n'
                "        # TODO: add heuristic prediction code here\n"
                "        # Example: return 1 if some feature exceeds a threshold, else 0\n"
                "        # return np.where(x['some_feature'] > threshold, 1, 0)\n"
                "        pass\n"
                "```\n"
                "\n"
                "## VARIABLES IN ENVIRONMENT:\n"
                "Your code has access to these variables:\n"
                "- `val_sample_df`: pandas DataFrame with validation sample\n"
                "- `task_analysis`: dict containing data analysis contextual to the ML task\n"
                "- `stats`: dict containing statistics for the dataset\n"
                "- `metric`: Metric we are optimising for in the broader ML project\n"
                "- `validate_baseline_predictor`: function for validating the predictor works"
                "\n"
                "## FUNCTIONS IN ENVIRONMENT:\n"
                "- `validate_baseline_predictor(predictor, name, description)`: Validate predictor works\n"
                "- `save_baseline_code(code)`: Save predictor code to disk\n"
                "\n"
                "## WORKFLOW:\n"
                "1. Analyze the broader task, `task_analysis` and `stats` to understand the problem\n"
                "2. Define a predictor class (heuristic ONLY)\n"
                "3. Instantiate and validate: validate_baseline_predictor(predictor, name, description)\n"
                "4. Once the predictor is successfully validated, save its code as string: save_baseline_code(code_string)\n"
                "5. Call final_answer() with rationale\n"
                "\n"
                "## CRITICAL:\n"
                "- Use task_analysis['output_targets'] to identify target column(s)\n"
                "- Predictor must have standard .predict(X) -> array interface\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                temperature=self.config.get_temperature("baseline_builder"),
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[
                get_save_baseline_code_tool(self.context, val_sample_df),
            ],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["collections", "pandas", "pandas.*", "numpy", "numpy.*"],
            max_steps=15,
        )

    @agent_span("BaselineBuilderAgent")
    def run(self) -> Baseline:
        """
        Build baseline predictor using agent.

        Returns:
            Baseline model with performance
        """

        logger.info("Building baseline predictor...")

        # Load samples to pandas
        val_sample_df = self.spark.read.parquet(self.context.val_sample_uri).toPandas()

        agent = self._build_agent(val_sample_df)

        task_type = self.context.task_analysis.get("task_type", "unknown")
        target_columns = self.context.output_targets
        metric_name = self.context.metric.name

        task = (
            f"Create a simple baseline predictor for this ML task.\n\n"
            f"**ML TASK**: {self.context.intent}\n\n"
            f"Task Type: {task_type}\n"
            f"Target Column(s): {target_columns}\n"
            f"Metric: {metric_name}\n\n"
            f"Build ONE simple baseline (heuristics preferred) that makes sense for this ML task. "
            f"Register it and evaluate performance using the tools provided."
        )

        agent.run(
            task=task,
            additional_args={
                "val_sample_df": val_sample_df,
                "task_analysis": self.context.task_analysis,
                "stats": self.context.stats,
                "metric": self.context.metric,
                "validate_baseline_predictor": get_validate_baseline_predictor_tool(self.context, val_sample_df),
            },
        )

        # Extract results from context
        if self.context.baseline_predictor is None:
            raise ValueError("Agent did not validate a baseline predictor")

        code_path = self.context.scratch.get("_baseline_code_path")
        if not code_path:
            raise ValueError("Agent did not save baseline code")

        # Performance already computed in validation tool (to let agent see metric errors)
        performance = self.context.baseline_performance
        if performance is None:
            # Fallback: compute if somehow not set
            performance = self._evaluate_performance(val_sample_df)
            self.context.baseline_performance = performance

        # Create Baseline object
        baseline_name = self.context.scratch.get("_baseline_name", "agent_baseline")
        baseline = Baseline(
            name=baseline_name,
            model_type="baseline",
            performance=self.context.baseline_performance,
            model_artifacts_path=self.context.work_dir / DirNames.BUILD_DIR / "search" / "baselines",
            metadata={
                "description": self.context.scratch.get("_baseline_description", ""),
                "predictor_code_file": f"{baseline_name}.py",
            },
        )

        logger.info(f"Baseline built successfully: {baseline.name} ({baseline.performance:.4f})")
        return baseline

    def _evaluate_performance(self, val_sample_df) -> float:
        """
        Evaluate baseline predictor performance on validation sample.

        Returns:
            Performance metric value
        """
        from plexe.helpers import compute_metric

        # Separate features from target
        target_cols = self.context.output_targets
        feature_cols = [col for col in val_sample_df.columns if col not in target_cols]

        X_val = val_sample_df[feature_cols]
        y_val = val_sample_df[target_cols[0]]

        # Make predictions
        y_pred = self.context.baseline_predictor.predict(X_val)

        # Compute metric
        performance = compute_metric(y_true=y_val.values, y_pred=y_pred, metric_name=self.context.metric.name)

        logger.info(f"Baseline performance: {self.context.metric.name}={performance:.4f}")

        return float(performance)
