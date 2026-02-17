"""
Metric Selector Agent.

Selects appropriate evaluation metric based on task analysis using smolagents.
"""

import logging

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.models import BuildContext, Metric
from plexe.config import Config, StandardMetric
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model
from plexe.tools.submission import get_submit_metric_choice_tool

logger = logging.getLogger(__name__)


class MetricSelectorAgent:
    """
    Agent that selects evaluation metric using smolagents with structured submission tool.

    Narrow responsibility: Choose appropriate metric for the ML task.
    """

    def __init__(self, context: BuildContext, config: Config):
        self.context = context
        self.config = config
        self.llm_model = config.metric_selection_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with metric submission tool."""
        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="MetricSelector",
            instructions=(
                "## YOUR ROLE:\n"
                "Select the most appropriate evaluation metric for this ML task.\n"
                "Your choice should reflect what 'good performance' means for THIS SPECIFIC TASK.\n"
                "\n"
                f"{feedback_section}"
                "## CONSIDER:\n"
                "1. **User Intent**: What does the user care about? (from intent description)\n"
                "2. **Task Type**: Classification vs regression\n"
                "3. **Class Balance**: Imbalanced → f1_score/roc_auc, balanced → accuracy\n"
                "4. **Data Challenges**: Any issues that affect metric choice?\n"
                "5. **Standard Practices**: What's commonly used for this task type?\n"
                "\n"
                "## INPUTS PROVIDED:\n"
                "- `task_analysis`: Dict with task_type, num_classes, data_challenges, target_analysis\n"
                "- `intent`: User's task description\n"
                "\n"
                "## AVAILABLE STANDARD METRICS:\n"
                f"{MetricSelectorAgent._format_available_metrics()}\n"
                "\n"
                "## TASK:\n"
                "1. Review task_analysis and intent\n"
                "2. Determine which metric best reflects success for THIS task\n"
                "3. Call submit_metric_choice(rationale, metric_name, optimization_direction)\n"
                "\n"
                "CRITICAL:\n"
                "- Prefer standard metrics when possible (built-in support)\n"
                "- Only use custom metric names for truly unique business needs\n"
                "- Your rationale should explain WHY this metric fits this specific task\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_submit_metric_choice_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports,
            max_steps=5,
        )

    @agent_span("MetricSelectorAgent")
    def run(self) -> Metric:
        """Run metric selection with structured submission."""

        logger.info("Selecting evaluation metric...")

        # Build agent
        agent = self._build_agent()

        # Extract task info
        task_type = self.context.task_analysis.get("task_type", "unknown")
        num_classes = self.context.task_analysis.get("num_classes", "N/A")
        data_challenges = self.context.task_analysis.get("data_challenges", [])
        target_analysis = self.context.task_analysis.get("target_analysis", {})

        # Build task prompt
        task = (
            f"Select the best evaluation metric for this task:\n"
            f"\n"
            f"**ML TASK**: {self.context.intent}\n"
            f"\n"
            f"Task Type: {task_type}\n"
            f"Number of Classes: {num_classes}\n"
            f"Data Challenges: {data_challenges}\n"
            f"Target Analysis: {target_analysis}\n"
            f"\n"
            f"Available standard metrics:\n"
            f"{MetricSelectorAgent._format_available_metrics()}\n"
            f"\n"
            f"Choose the metric that best reflects model quality for this specific task."
        )

        # Run agent
        agent.run(
            task=task,
            additional_args={
                "task_analysis": self.context.task_analysis,
                "intent": self.context.intent,
            },
        )

        # Retrieve metric from context (saved by tool)
        if not self.context.metric:
            raise ValueError("Agent did not submit metric choice")

        metric = self.context.metric
        selection = self.context.scratch.get("_metric_selection", {})

        logger.info(
            f"Selected metric: {metric.name} ({metric.optimization_direction})\n"
            f"Rationale: {selection.get('rationale', 'N/A')}"
        )

        return metric

    @staticmethod
    def _format_available_metrics() -> str:
        """Generate formatted list of available standard metrics from StandardMetric enum."""
        classification_metrics = []
        regression_metrics = []

        for metric in StandardMetric:
            metric_value = metric.value

            # Categorize metrics based on their names
            # Regression metrics
            if any(
                key in metric_value
                for key in [
                    "mse",
                    "rmse",
                    "mae",
                    "r2",
                    "mape",
                    "median_absolute_error",
                    "max_error",
                    "explained_variance",
                ]
            ):
                regression_metrics.append(metric_value)
            # Classification metrics (everything else)
            else:
                classification_metrics.append(metric_value)

        # Format output
        output = "**Classification:**\n"
        output += "- " + ", ".join(sorted(classification_metrics)) + "\n\n"
        output += "**Regression:**\n"
        output += "- " + ", ".join(sorted(regression_metrics)) + "\n"

        return output
