"""
Insight Extractor Agent.

Analyzes experiment results to extract structured learnings.
"""

import logging

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.models import BuildContext, Hypothesis, Solution
from plexe.config import Config
from plexe.tools.submission import get_save_insight_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class InsightExtractorAgent:
    """
    Agent that extracts structured insights from experiment results.

    Analyzes N variant solutions to understand:
    - What changed and what effect it had
    - Which variations worked best
    - Under what conditions findings apply
    - Whether findings contradict or refine existing insights
    """

    def __init__(
        self,
        hypothesis: Hypothesis,
        variant_solutions: list[Solution],
        insight_store,  # InsightStore instance
        context: BuildContext,
        config: Config,
    ):
        """
        Initialize agent.

        Args:
            hypothesis: Hypothesis that was tested
            variant_solutions: List of solutions created from hypothesis
            insight_store: InsightStore for saving insights
            context: Build context
            config: Configuration
        """
        self.hypothesis = hypothesis
        self.variant_solutions = variant_solutions
        self.insight_store = insight_store
        self.context = context
        self.config = config
        self.llm_model = config.insight_extractor_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with insight submission tool."""

        # Get parent node for comparison
        parent_node = self._get_parent_node()
        parent_perf = parent_node.performance if parent_node and parent_node.performance else None

        # Summarize variant results with comparison to parent
        results_summary = self._summarize_results(parent_perf)

        # Get existing insights for context
        existing_insights = self._summarize_existing_insights(self.insight_store)

        # Get dataset context
        dataset_context = self._get_dataset_context()

        instructions = (
            "## YOUR ROLE:\n"
            "You are an ML scientist analyzing experiment results to extract structured learnings.\n"
            "\n"
            "## ML TASK:\n"
            f"Goal: {self.context.intent}\n"
            f"Metric: {self.context.metric.name} ({self.context.metric.optimization_direction} is better)\n"
            "\n"
            f"## DATASET:\n"
            f"{dataset_context}\n"
            "\n"
            "## HYPOTHESIS TESTED:\n"
            f"Varied: {self.hypothesis.vary}\n"
            f"Rationale: {self.hypothesis.rationale}\n"
            f"Expected Impact: {self.hypothesis.expected_impact}\n"
            "\n"
            "## VARIANT RESULTS:\n"
            f"{results_summary}\n"
            "\n"
            "## EXISTING INSIGHTS:\n"
            f"{existing_insights}\n"
            "\n"
            "## TASK:\n"
            "Analyze the variant results and extract insights.\n"
            "\n"
            "For each meaningful finding, call save_insight() with:\n"
            "- change: What was varied (e.g., 'n_estimators: 100→400')\n"
            "- effect: Observed outcome (e.g., '+5.8% improvement, peak at 250, diminishing returns after')\n"
            "- context_str: When this applies (e.g., 'for binary classification with ~8k rows, ~13 features')\n"
            "- confidence: 'high' (all variants agree), 'medium' (some noise), 'low' (unclear pattern)\n"
            "- supporting_evidence: List of solution IDs that support this\n"
            "\n"
            "Guidelines:\n"
            "1. Extract 1-3 insights maximum (focus on most important learnings)\n"
            "2. Be specific about what changed and the quantitative effect\n"
            "3. Qualify insights with context (data characteristics, task type)\n"
            "4. If all variants failed, extract insight about WHY (common error pattern)\n"
            "5. If findings contradict existing insights, note that in your analysis\n"
            "\n"
            "Example:\n"
            "save_insight(\n"
            "    change='n_estimators increased from 100 to 250',\n"
            "    effect='+5.8% improvement (0.82→0.87), diminishing returns beyond 250',\n"
            "    context_str='for binary classification on ~8k row dataset with 13 features',\n"
            "    confidence='high',\n"
            "    supporting_evidence=[33, 34, 35]\n"
            ")\n"
        )

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="InsightExtractor",
            instructions=instructions,
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
                reasoning_effort="minimal",
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_insight_tool(self.insight_store)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports,
            max_steps=15,
        )

    @agent_span("InsightExtractorAgent")
    def run(self) -> int:
        """
        Extract insights from variant results.

        Returns:
            Number of insights extracted
        """
        logger.info(f"Extracting insights from {len(self.variant_solutions)} variant results...")

        # Get current insight count
        initial_count = len(self.insight_store)

        agent = self._build_agent()

        # Run agent
        agent.run(
            task="Analyze the variant results and extract structured insights.",
            additional_args={},
        )

        # Count new insights
        insights_added = len(self.insight_store) - initial_count

        logger.info(f"Extracted {insights_added} new insights")

        return insights_added

    def _get_parent_node(self) -> Solution | None:
        """Get parent node from first variant solution."""
        if not self.variant_solutions:
            return None
        first_variant = self.variant_solutions[0]
        return first_variant.parent if first_variant else None

    def _get_dataset_context(self) -> str:
        """Get dataset context."""
        stats = self.context.stats or {}
        task = self.context.task_analysis or {}

        return (
            f"{stats.get('total_rows', '?')} rows, {stats.get('total_columns', '?')} columns, "
            f"{task.get('num_classes', '?')} classes"
        )

    def _summarize_results(self, parent_perf: float | None) -> str:
        """Generate summary of variant results with comparison to parent."""
        if not self.variant_solutions:
            return "No variant results available.\n"

        summary = ""

        # Show parent performance for comparison
        if parent_perf is not None:
            summary += f"Parent Performance: {parent_perf:.4f}\n\n"

        for sol in self.variant_solutions:
            variant_id = sol.plan.variant_id if sol.plan else "?"
            status = "SUCCESS" if sol.is_successful else f"FAILED ({sol.error[:50]})" if sol.is_buggy else "PENDING"

            summary += f"  Variant {variant_id} (solution {sol.solution_id}):\n"

            if sol.plan:
                summary += f"    Change: {sol.plan.model.change_summary}\n"

            if sol.is_successful:
                perf_delta = (sol.performance - parent_perf) if parent_perf is not None else None
                perf_str = f"{sol.performance:.4f}"
                if perf_delta is not None:
                    perf_pct = (perf_delta / parent_perf * 100) if parent_perf else 0
                    perf_str += f" ({perf_delta:+.4f}, {perf_pct:+.1f}%)"
                summary += f"    Performance: {perf_str}\n"
            else:
                summary += f"    Status: {status}\n"

        return summary

    @staticmethod
    def _summarize_existing_insights(insight_store) -> str:
        """Generate summary of existing insights for context."""
        if not insight_store or len(insight_store) == 0:
            return "No existing insights.\n"

        active = insight_store.get_all()
        if not active:
            return "No active insights.\n"

        summary = "Existing insights (check if new findings refine or contradict these):\n"
        for insight in active:
            summary += f"  #{insight.id}: {insight.change} → {insight.effect}\n"

        return summary
