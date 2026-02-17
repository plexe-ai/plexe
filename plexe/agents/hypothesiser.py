"""
Hypothesiser Agent.

Generates strategic hypotheses for next exploration based on insights and search history.
"""

import logging

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.models import BuildContext, Hypothesis
from plexe.config import Config, get_routing_for_model
from plexe.tools.submission import get_save_hypothesis_tool
from plexe.utils.tracing import agent_span
from plexe.search.journal import SearchJournal

logger = logging.getLogger(__name__)


class HypothesiserAgent:
    """
    Agent that generates hypotheses for next search direction.

    Analyzes search tree, insights, and history to decide:
    - Which node to expand
    - What to vary (features/model/both)
    - How many variants to try
    - What to keep from parent
    """

    def __init__(
        self,
        journal: SearchJournal,
        context: BuildContext,
        config: Config,
        expand_solution_id: int,  # Policy tells us which solution to expand
    ):
        """
        Initialize agent.

        Args:
            journal: Search journal with history
            context: Build context
            config: Configuration
            expand_solution_id: Which solution to expand (from policy)
        """
        self.journal = journal
        self.context = context
        self.config = config
        self.expand_solution_id = expand_solution_id
        self.llm_model = config.hypothesiser_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with hypothesis submission tool."""

        # Get node to expand (may be None if first iteration or invalid ID)
        expand_node = (
            next((n for n in self.journal.nodes if n.solution_id == self.expand_solution_id), None)
            if self.journal.nodes
            else None
        )
        node_summary = self._summarize_node(expand_node) if expand_node else "No parent node (starting from scratch)"

        # Get insights
        insights_summary = (
            self._summarize_insights(self.context.insight_store) if self.context.insight_store else "No insights yet"
        )

        # Get search history
        history = self.journal.get_history(limit=10)
        history_summary = self._summarize_history(history)

        # Get dataset and task context
        dataset_context = self._get_dataset_context()

        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        instructions = (
            "## YOUR ROLE:\n"
            "You are an ML scientist designing experiments to improve model performance.\n"
            "Your task is to generate a strategic hypothesis for the next search direction.\n"
            "\n"
            f"{feedback_section}"
            "## ML TASK:\n"
            f"Goal: {self.context.intent}\n"
            f"Task Type: {self.context.task_analysis.get('task_type', 'unknown')}\n"
            f"Target: {', '.join(self.context.output_targets)}\n"
            f"Metric: {self.context.metric.name} ({self.context.metric.optimization_direction} is better)\n"
            f"Baseline Performance: {self.journal.baseline_performance:.4f}\n"
            f"Best So Far: {self.journal.best_performance:.4f}\n"
            "\n"
            f"## DATASET CONTEXT:\n"
            f"{dataset_context}\n"
            "\n"
            f"## NODE TO EXPAND:\n"
            f"Solution #{self.expand_solution_id}:\n"
            f"{node_summary}\n"
            "\n"
            "## ACCUMULATED INSIGHTS:\n"
            f"{insights_summary}\n"
            "\n"
            "## RECENT SEARCH HISTORY:\n"
            f"{history_summary}\n"
            "\n"
            "## YOUR TASK:\n"
            "Generate ONE hypothesis for what to try next.\n"
            "\n"
            "Consider:\n"
            "1. What have we learned? (insights)\n"
            "2. What's been tried? (history)\n"
            "3. What shows promise vs diminishing returns?\n"
            "4. Focus on features, model, or both?\n"
            "5. How many variants? (1 for targeted, 3-5 for exploration)\n"
            "\n"
            "Call save_hypothesis() with your decision, then call final_answer() to complete the task.\n"
        )

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="Hypothesiser",
            instructions=instructions,
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
                reasoning_effort="minimal",
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_hypothesis_tool(self.context, self.expand_solution_id)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports,
            max_steps=3,
        )

    @agent_span("HypothesiserAgent")
    def run(self) -> Hypothesis:
        """
        Generate hypothesis for next exploration.

        Returns:
            Hypothesis object
        """
        logger.info(f"Generating hypothesis for expanding solution {self.expand_solution_id}...")

        agent = self._build_agent()

        # Run agent
        agent.run(
            task="Analyze the search state and generate a hypothesis for what to explore next.",
            additional_args={},
        )

        # Retrieve hypothesis
        hypothesis = self.context.scratch.get("_hypothesis")

        if not hypothesis:
            raise ValueError("Agent did not save hypothesis")

        logger.info(
            f"Hypothesis generated: {hypothesis.focus} - vary {hypothesis.vary}, {hypothesis.num_variants} variants"
        )

        return hypothesis

    def _get_dataset_context(self) -> str:
        """Get dataset and task context for informed decision-making."""
        stats = self.context.stats or {}
        task = self.context.task_analysis or {}

        rows = stats.get("total_rows", "unknown")
        cols = stats.get("total_columns", "unknown")
        num_classes = task.get("num_classes", "N/A")

        return (
            f"Dataset: {rows} rows, {cols} columns\n"
            f"Classes: {num_classes}\n"
            f"Features: {len(task.get('feature_columns', []))} available\n"
        )

    @staticmethod
    def _summarize_node(node) -> str:
        """Generate summary of a solution node."""
        if not node:
            return "No node"

        summary = f"  Performance: {node.performance:.4f}\n" if node.performance else "  Performance: N/A\n"

        if node.plan:
            summary += f"  Features: {node.plan.features.strategy}\n"
            summary += f"  Model: {node.plan.model.change_summary or node.plan.model.directive}\n"
        else:
            summary += "  (No structured plan - legacy solution)\n"

        if node.error:
            summary += f"  Error: {node.error[:100]}\n"

        return summary

    @staticmethod
    def _summarize_insights(insight_store) -> str:
        """Generate summary of active insights."""
        if not insight_store or len(insight_store) == 0:
            return "No insights accumulated yet.\n"

        active = insight_store.get_all()  # Can filter to get_active() if needed
        if not active:
            return "No active insights.\n"

        summary = ""
        for insight in active[:10]:  # Show up to 10 most recent
            summary += f"  Insight #{insight.id}: {insight.change} → {insight.effect}\n"
            summary += f"    Context: {insight.context}\n"
            summary += f"    Confidence: {insight.confidence}\n"

        return summary

    @staticmethod
    def _summarize_history(history: list[dict]) -> str:
        """Generate summary of recent search attempts."""
        if not history:
            return "No search history yet.\n"

        summary = ""
        for entry in history:
            solution_id = entry["solution_id"]
            stage = entry["stage"]
            success = entry["success"]
            perf = entry.get("performance")

            status = f"✓ {perf:.4f}" if success and perf else ("✗ FAILED" if not success else "pending")
            summary += f"  Solution {solution_id} ({stage}): {status}\n"

            if entry.get("error"):
                summary += f"    Error: {entry['error'][:80]}\n"

        return summary
