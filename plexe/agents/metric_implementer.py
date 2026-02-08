"""
Metric Implementation Agent.

Generates metric computation function code.
"""

import logging
from typing import Any

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.models import BuildContext
from plexe.config import Config
from plexe.tools.submission import get_save_metric_implementation_fn
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class MetricImplementationAgent:
    """
    Agent that generates metric computation function code.

    Generates a compute_metric(y_true, y_pred) function that will be used
    consistently across baseline evaluation, model selection, and final testing.
    """

    def __init__(self, context: BuildContext, config: Config):
        """
        Initialize agent.

        Args:
            context: Build context with metric definition
            config: Configuration
        """
        self.context = context
        self.config = config
        self.llm_model = config.metric_selection_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with metric implementation tool."""
        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="MetricImplementer",
            instructions=(
                "## YOUR ROLE:\n"
                "Create a Python function that computes the evaluation metric.\n"
                "This function will be used for ALL evaluations (baseline, models, final test).\n"
                "\n"
                "## CONTEXT:\n"
                "Available in `additional_args`:\n"
                "- `metric`: Metric object with name and optimization_direction\n"
                "- `task_analysis`: Task details (task_type, num_classes, etc.)\n"
                "\n"
                "## YOUR TASK:\n"
                "1. Write a function called `compute_metric(y_true, y_pred)` that computes the metric\n"
                "2. Execute your code to define the function\n"
                "3. Call save_metric_implementation(compute_metric) with the FUNCTION OBJECT\n"
                "\n"
                "## SUBMISSION FUNCTION:\n"
                "You have access to `save_metric_implementation` function in your environment.\n"
                "Call it with the function object (not a string):\n"
                "  save_metric_implementation(compute_metric)\n"
                "\n"
                "## GUIDELINES:\n"
                "**Classification:** accuracy, f1_score (with average='weighted'), roc_auc, precision, recall\n"
                "**Regression:** rmse, mae, r2_score, mape\n"
                "\n"
                "## EXAMPLE:\n"
                "```python\n"
                "# Define function\n"
                "def compute_metric(y_true, y_pred):\n"
                "    from sklearn.metrics import f1_score\n"
                "    return f1_score(y_true, y_pred, average='weighted')\n"
                "\n"
                "# Submit the function object\n"
                "result = save_metric_implementation(compute_metric)\n"
                "print(result)  # Check if successful\n"
                "```\n"
                "\n"
                "## REQUIREMENTS:\n"
                "- Function named `compute_metric`\n"
                "- Takes exactly 2 args: y_true, y_pred\n"
                "- Returns single float\n"
                "- Imports inside function for portability\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[],  # No tools - submission via additional_args
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["sklearn", "sklearn.*", "sklearn.metrics", "numpy", "numpy.*"],
            max_steps=10,
        )

    @agent_span("MetricImplementationAgent")
    def run(self) -> Any:
        """
        Generate metric computation function.

        Returns:
            Metric computation function (callable)
        """

        logger.info(f"Generating implementation for metric: {self.context.metric.name}")

        # Build agent
        agent = self._build_agent()

        # Build task
        metric = self.context.metric
        task_type = self.context.task_analysis.get("task_type", "unknown")
        num_classes = self.context.task_analysis.get("num_classes", 0)

        task = (
            f"Generate a compute_metric(y_true, y_pred) function.\n"
            f"\n"
            f"Metric: {metric.name}\n"
            f"Optimization: {metric.optimization_direction} is better\n"
            f"Task Type: {task_type}\n"
            f"Number of Classes: {num_classes}\n"
            f"\n"
            f"The function will be used to evaluate:\n"
            f"- Baseline predictor\n"
            f"- Models during search\n"
            f"- Final test set evaluation\n"
            f"\n"
            f"Ensure the function:\n"
            f"1. Uses appropriate sklearn.metrics for '{metric.name}'\n"
            f"2. Handles multiclass with proper averaging if needed\n"
            f"3. Returns higher/lower values correctly based on optimization direction\n"
            f"4. Is robust to edge cases\n"
            f"\n"
            f"Call save_metric_implementation(compute_metric) with your function object.\n"
        )

        # Run agent
        try:
            agent.run(
                task=task,
                additional_args={
                    "metric": metric,
                    "task_analysis": self.context.task_analysis,
                    "save_metric_implementation": get_save_metric_implementation_fn(self.context),
                },
            )

            # Retrieve function from context
            compute_metric_func = self.context.compute_metric

            if not compute_metric_func:
                raise ValueError("Agent did not save metric implementation function")

            logger.info(f"Metric implementation generated for {metric.name}")

            return compute_metric_func

        except Exception as e:
            logger.error(f"Metric implementation generation failed: {e}")
            raise
