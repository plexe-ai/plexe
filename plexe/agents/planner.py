"""
Planner Agent.

Creates concrete plan specifications from hypotheses.
"""

import logging

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

from plexe.models import BuildContext, Hypothesis, UnifiedPlan
from plexe.config import Config
from plexe.tools.submission import get_save_plan_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model
from plexe.search.journal import SearchJournal

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Agent that creates concrete plan specifications from hypotheses.

    Takes a Hypothesis and generates N concrete UnifiedPlan variants
    with specific hyperparameters, strategies, and rationales.
    """

    def __init__(
        self,
        journal: SearchJournal,
        context: BuildContext,
        config: Config,
        hypothesis: Hypothesis | None = None,
        num_bootstrap: int = 1,
    ):
        """
        Initialize agent.

        Args:
            journal: Search journal for parent node lookup
            context: Build context
            config: Configuration
            hypothesis: Strategic hypothesis to implement (None for bootstrap mode)
            num_bootstrap: Number of diverse solutions to create in bootstrap mode
        """
        self.hypothesis = hypothesis
        self.journal = journal
        self.context = context
        self.config = config
        self.num_bootstrap = num_bootstrap
        self.llm_model = config.planner_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with plan submission tool."""

        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Get dataset context
        dataset_context = self._get_dataset_context()

        # Bootstrap mode vs hypothesis-driven mode
        if self.hypothesis is None:
            # Bootstrap: Create diverse from-scratch solutions
            variant_ids = ["A", "B", "C", "D", "E"][: self.num_bootstrap]
            num_variants = self.num_bootstrap

            instructions = (
                "## YOUR ROLE:\n"
                "You are an ML experiment designer creating diverse initial solution strategies.\n"
                f"Generate {num_variants} DIVERSE plan(s) from scratch to bootstrap the search.\n"
                "\n"
                f"{feedback_section}"
                "## ML TASK:\n"
                f"Goal: {self.context.intent}\n"
                f"Metric: {self.context.metric.name} ({self.context.metric.optimization_direction} is better)\n"
                "\n"
                f"## DATASET:\n"
                f"{dataset_context}\n"
                "\n"
                "## AVAILABLE MODEL ARCHITECTURES:\n"
                f"{self._get_model_types_guidance()}\n"
                "\n"
                "## TASK:\n"
                f"Create {num_variants} DIVERSE initial solution plan(s) (IDs: {', '.join(variant_ids)}).\n"
                "Each plan should explore different approaches to feature engineering and modeling.\n"
                "\n"
                "For EACH plan, call save_plan() with:\n"
                "- variant_id: Variant letter (A, B, C, etc.)\n"
                f"- model_type: Choose from: {', '.join(self.context.viable_model_types)}\n"
                "- feature_strategy: Use 'new' (creating from scratch)\n"
                "- feature_changes: {} (empty dict for new pipelines)\n"
                "- feature_rationale: What feature engineering approach to try\n"
                "- model_directive: Natural language instruction for model architecture and training\n"
                f"{self._get_model_directive_examples()}\n"
                "- model_change_summary: Brief summary (e.g., 'XGBoost: n_estimators=150' or 'CatBoost: iterations=500')\n"
                "- model_rationale: Why this architecture/configuration\n"
                "\n"
                "Make each variant meaningfully different.\n"
                "After saving all plans, call final_answer() to complete the task.\n"
            )

            # Create dummy hypothesis for tool factory
            dummy_hypothesis = Hypothesis(
                expand_solution_id=-1,
                focus="both",
                vary="bootstrap",
                num_variants=num_variants,
                rationale="Bootstrap",
                keep_from_parent=[],
                expected_impact="Initial diversity",
            )
            tool_hypothesis = dummy_hypothesis

        else:
            # Normal hypothesis-driven mode
            parent_node = next(
                (n for n in self.journal.nodes if n.solution_id == self.hypothesis.expand_solution_id), None
            )
            parent_summary = self._summarize_parent(parent_node) if parent_node else "Parent not found"

            insights_summary = (
                self._summarize_insights(self.context.insight_store) if self.context.insight_store else "No insights"
            )

            variant_ids = ["A", "B", "C", "D", "E"][: self.hypothesis.num_variants]
            num_variants = self.hypothesis.num_variants

            # Get parent's model type for guidance
            parent_model_type = parent_node.model_type if parent_node else "xgboost"

            instructions = (
                "## YOUR ROLE:\n"
                "You are an ML experiment designer creating concrete solution specifications.\n"
                f"Generate {num_variants} plan variants implementing the hypothesis below.\n"
                "\n"
                f"{feedback_section}"
                "## ML TASK:\n"
                f"Goal: {self.context.intent}\n"
                f"Metric: {self.context.metric.name} ({self.context.metric.optimization_direction} is better)\n"
                "\n"
                f"## DATASET:\n"
                f"{dataset_context}\n"
                "\n"
                "## AVAILABLE MODEL ARCHITECTURES:\n"
                f"{self._get_model_types_guidance()}\n"
                "\n"
                "## HYPOTHESIS:\n"
                f"Vary: {self.hypothesis.vary}\n"
                f"Rationale: {self.hypothesis.rationale}\n"
                f"Keep from Parent: {self.hypothesis.keep_from_parent}\n"
                "\n"
                "## PARENT SOLUTION:\n"
                f"#{self.hypothesis.expand_solution_id}:\n"
                f"{parent_summary}\n"
                "\n"
                "## RELEVANT INSIGHTS:\n"
                f"{insights_summary}\n"
                "\n"
                "## TASK:\n"
                f"Create {num_variants} variants (IDs: {', '.join(variant_ids)}).\n"
                f"Vary '{self.hypothesis.vary}' across variants with different values.\n"
                "\n"
                "For each variant, call save_plan() with:\n"
                "- variant_id: Variant letter\n"
                f"- model_type: Typically '{parent_model_type}' (parent's type), unless hypothesis is about trying a different architecture\n"
                f"  Available options: {', '.join(self.context.viable_model_types)}\n"
                "- feature_strategy, feature_changes, feature_rationale: Based on hypothesis focus\n"
                "- model_directive: Natural language instruction for model architecture and training config\n"
                "- model_change_summary: What changed (e.g., 'n_estimators: 100→~250' or 'iterations: 500→800')\n"
                "- model_rationale: Why this change\n"
                "\n"
                "The directive guides the ModelDefinerAgent - it's NOT executable code.\n"
                "After saving all plans, call final_answer() to complete the task.\n"
            )
            tool_hypothesis = self.hypothesis

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="Planner",
            instructions=instructions,
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
                reasoning_effort="minimal",
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_plan_tool(self.context, tool_hypothesis, self.context.viable_model_types)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports,
            max_steps=5,
        )

    @agent_span("PlannerAgent")
    def run(self) -> list[UnifiedPlan]:
        """
        Generate concrete plan specifications.

        Returns:
            List of UnifiedPlan objects (one per variant)
        """
        expected_count = self.hypothesis.num_variants if self.hypothesis else self.num_bootstrap
        mode = "bootstrap" if self.hypothesis is None else "hypothesis-driven"

        logger.info(f"Generating {expected_count} plan variant(s) ({mode} mode)...")

        agent = self._build_agent()

        # Clear plans from context
        self.context.scratch["_plans"] = []

        # Run agent
        task_desc = (
            f"Create {expected_count} diverse initial solution plan(s) from scratch"
            if self.hypothesis is None
            else f"Create {expected_count} plan variant(s) based on the hypothesis"
        )
        agent.run(task=task_desc, additional_args={})

        # Retrieve plans
        plans = self.context.scratch.get("_plans", [])

        if len(plans) != expected_count:
            logger.warning(f"Expected {expected_count} plans, got {len(plans)}. Proceeding with available plans.")

        if not plans:
            raise ValueError("Agent did not generate any plans")

        logger.info(f"Generated {len(plans)} plan variant(s)")

        return plans

    def _get_dataset_context(self) -> str:
        """Get dataset context."""
        stats = self.context.stats or {}
        task = self.context.task_analysis or {}

        return (
            f"{stats.get('total_rows', '?')} rows, {stats.get('total_columns', '?')} columns\n"
            f"{len(task.get('feature_columns', []))} features, {task.get('num_classes', '?')} classes\n"
        )

    @staticmethod
    def _summarize_parent(node) -> str:
        """Generate detailed summary of parent node."""
        if not node:
            return "No parent node"

        summary = f"  Performance: {node.performance:.4f}\n" if node.performance else "  Performance: N/A\n"

        if node.plan:
            # Show feature configuration
            summary += "  Features:\n"
            summary += f"    Strategy: {node.plan.features.strategy}\n"
            if node.plan.features.changes:
                summary += f"    Changes: {node.plan.features.changes}\n"

            # Show model configuration
            summary += "  Model:\n"
            summary += f"    Type: {node.plan.model.model_type}\n"
            summary += f"    Directive: {node.plan.model.directive[:80] if node.plan.model.directive else 'N/A'}\n"
            summary += f"    Change: {node.plan.model.change_summary}\n"
        else:
            summary += "  (Legacy node - no structured plan)\n"

        return summary

    @staticmethod
    def _summarize_insights(insight_store) -> str:
        """Generate summary of relevant insights."""
        if not insight_store or len(insight_store) == 0:
            return "No insights available.\n"

        active = insight_store.get_all()
        if not active:
            return "No active insights.\n"

        summary = ""
        for insight in active[:5]:  # Show up to 5
            summary += f"  #{insight.id}: {insight.change} → {insight.effect} ({insight.confidence} confidence)\n"

        return summary

    def _get_model_types_guidance(self) -> str:
        """Generate model types guidance based on viable_model_types from context."""
        # Use viable_model_types computed in Phase 1 (already filtered for task compatibility)
        viable = self.context.viable_model_types if self.context.viable_model_types else ["xgboost"]

        guidance = f"**AVAILABLE MODEL TYPES**: You can choose from: {', '.join(viable)}\n\n"

        if "xgboost" in viable:
            guidance += "- **xgboost**: Gradient boosted decision trees\n"
            guidance += "  * Params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree\n"

        if "catboost" in viable:
            guidance += "- **catboost**: Gradient boosted decision trees with native categorical support\n"
            guidance += "  * Params: iterations, depth, learning_rate, l2_leaf_reg\n"
            guidance += "  * NOTE: CatBoost handles categorical features automatically\n"

        if "lightgbm" in viable:
            guidance += "- **lightgbm**: Fast gradient boosted decision trees with leaf-wise growth\n"
            guidance += "  * Params: n_estimators, num_leaves, max_depth, learning_rate, subsample, colsample_bytree\n"
            guidance += "  * NOTE: num_leaves controls complexity (default 31), often faster than XGBoost/CatBoost\n"

        if "keras" in viable:
            guidance += "- **keras**: Neural networks (TensorFlow backend)\n"
            guidance += "  * Params: layer sizes, activation, dropout, optimizer, loss\n"
            guidance += f"  * Default epochs: {self.config.nn_default_epochs}\n"
            guidance += f"  * Max epochs: {self.config.nn_max_epochs}\n"

        if "pytorch" in viable:
            guidance += "- **pytorch**: Neural networks (PyTorch)\n"
            guidance += "  * Params: layer sizes, activation, dropout, optimizer, lr\n"
            guidance += f"  * Default epochs: {self.config.nn_default_epochs}\n"
            guidance += f"  * Max epochs: {self.config.nn_max_epochs}\n"

        if len(viable) > 1:
            guidance += "\n**FRAMEWORK EXPLORATION**: You can experiment with different model types across variants.\n"
            guidance += "For example, try CatBoost vs XGBoost to see which performs better for this dataset.\n"

        return guidance

    def _get_model_directive_examples(self) -> str:
        """Generate model directive examples based on viable_model_types."""
        viable = self.context.viable_model_types if self.context.viable_model_types else ["xgboost"]

        examples = []
        if "xgboost" in viable:
            examples.append(
                "    * XGBoost example: 'Create XGBoost with n_estimators around 150, learning_rate 0.1, max_depth 5'"
            )
        if "catboost" in viable:
            examples.append(
                "    * CatBoost example: 'Create CatBoost with iterations around 500, depth 6, learning_rate 0.05'"
            )
        if "lightgbm" in viable:
            examples.append(
                "    * LightGBM example: 'Create LightGBM with n_estimators around 200, num_leaves 31, learning_rate 0.05'"
            )
        if "keras" in viable:
            examples.append(
                "    * Keras example: 'Create 3-layer network (64, 32, 16 units, relu, dropout 0.2). Train 40 epochs, batch_size 64'"
            )
        if "pytorch" in viable:
            examples.append(
                "    * PyTorch example: 'Create 3-layer network (128, 64, 32 units, ReLU, dropout 0.3). Train 30 epochs, batch_size 64'"
            )

        return "\n".join(examples) if examples else "    * No examples available"
