"""
Feature Processor Agent.

Designs sklearn Pipeline for feature engineering based on plan specification.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline

from plexe.constants import ScratchKeys
from plexe.models import BuildContext
from plexe.config import Config
from plexe.tools.submission import get_save_pipeline_code_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class FeatureProcessorAgent:
    """
    Agent that designs sklearn Pipeline for feature engineering.

    Implements FeaturePlan specifications by generating sklearn Pipeline code.
    """

    def __init__(
        self,
        spark: SparkSession,
        train_uri: str,
        context: BuildContext,
        config: Config,
        plan: Any,  # FeaturePlan specification
    ):
        """
        Initialize agent.

        Args:
            spark: SparkSession for data access
            train_uri: Training data URI (for loading sample)
            context: Build context with stats and task analysis
            config: Configuration
            plan: FeaturePlan specification (strategy, changes, rationale)
        """
        self.spark = spark
        self.train_uri = train_uri
        self.context = context
        self.config = config
        self.llm_model = config.feature_processor_llm
        self.plan = plan

    def _build_agent(self, train_sample_df, val_sample_df) -> CodeAgent:
        """Build CodeAgent with pipeline submission function."""

        # Build stage-specific instructions
        instructions = self._build_instructions()

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="FeatureProcessor",
            instructions=instructions,
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_pipeline_code_tool(self.context, train_sample_df, val_sample_df)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["pandas", "pandas.*", "numpy", "numpy.*", "sklearn", "sklearn.*"],
            max_steps=20,
        )

    # TODO(IMAGE_TEXT_SUPPORT): Branch on context.data_layout here
    # See /IMAGE_TEXT_SUPPORT.md section 2
    # Currently: always returns Pipeline (tabular only)
    # Needed: return ImagePreprocessingConfig or TextPreprocessingConfig for non-tabular

    @agent_span("FeatureProcessorAgent")
    def run(self) -> tuple[Pipeline, str]:
        """
        Design feature engineering pipeline.

        Returns:
            (pipeline, reasoning) - sklearn Pipeline and natural language plan

        TODO: Return transformed data schema alongside pipeline
            Current: return (pipeline, reasoning)
            Future: return (pipeline, output_schema, reasoning)
            Where output_schema contains:
            - columns: List[str] with MEANINGFUL names (e.g., "HomePlanet_Earth", "Age_scaled")
            - dtypes: Dict[str, str] mapping column → data type
            - num_features: int
            - feature_mapping: Dict mapping output columns back to original columns
            This would enable interpretability, feature importance, and debugging.
        """

        logger.info("Designing feature pipeline from plan...")

        # ============================================
        # Step 1: Load Samples for Validation
        # ============================================
        # Load representative subset for validation (1000 rows balances coverage and speed)
        train_sample_df = self.spark.read.parquet(self.train_uri).limit(1000).toPandas()

        # Drop target AND group columns (ranking metadata, not features)
        columns_to_drop = list(self.context.output_targets)
        if self.context.group_column and self.context.group_column in train_sample_df.columns:
            columns_to_drop.append(self.context.group_column)

        train_features_df = train_sample_df.drop(columns=columns_to_drop, errors="ignore")

        # Load val sample for consistency validation
        val_sample_df = self.spark.read.parquet(self.context.val_sample_uri).limit(1000).toPandas()

        columns_to_drop_val = list(self.context.output_targets)
        if self.context.group_column and self.context.group_column in val_sample_df.columns:
            columns_to_drop_val.append(self.context.group_column)

        val_features_df = val_sample_df.drop(columns=columns_to_drop_val, errors="ignore")

        # ============================================
        # Step 2: Build Agent
        # ============================================
        agent = self._build_agent(train_features_df, val_features_df)

        # ============================================
        # Step 3: Run Agent
        # ============================================
        additional_args = {
            "stats_report": self.context.stats,
            "task_analysis": self.context.task_analysis,
            "sample_df": train_features_df,
        }

        try:
            result = agent.run(task="Implement the feature engineering plan", additional_args=additional_args)

            # Retrieve saved pipeline
            pipeline = self.context.scratch.get(ScratchKeys.SAVED_PIPELINE)

            if not pipeline:
                raise ValueError("Agent did not save pipeline")

            # Extract reasoning from agent output
            reasoning = str(result) if result else ""

            logger.info("Feature pipeline created successfully")
            return pipeline, reasoning

        except Exception as e:
            logger.error(f"Feature processing failed: {e}")
            raise

    def _build_instructions(self) -> str:
        """Build instructions for implementing the feature plan."""

        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Load pipeline template
        from pathlib import Path
        import textwrap

        template_path = Path(__file__).parent.parent / "templates" / "features" / "pipeline.tmpl.py"
        with open(template_path) as f:
            pipeline_template = textwrap.dedent(f.read()).strip()

        instructions = (
            "## YOUR ROLE:\n"
            "Design an sklearn Pipeline for feature engineering that transforms RAW input data.\n"
            "\n"
            f"{feedback_section}"
            "## PLAN TO IMPLEMENT:\n"
            f"Strategy: {self.plan.strategy}\n"
            f"Rationale: {self.plan.rationale}\n"
            f"Specific Changes: {self.plan.changes if self.plan.changes else 'None - create from scratch'}\n"
            "\n"
            f"## ML TASK (for context):\n"
            f"Goal: {self.context.intent}\n"
            f"Task Type: {self.context.task_analysis.get('task_type', 'unknown')}\n"
            f"Target Column(s): {', '.join(self.context.output_targets)}\n"
            "\n"
            "## DATA AVAILABLE:\n"
            f"- `sample_df`: pandas DataFrame of RAW training data (target columns {self.context.output_targets} have been removed)\n"
            "- `stats_report`: Statistical profiling (may reference target column, but it's NOT in sample_df)\n"
            "- `task_analysis`: ML task analysis (feature recommendations, task type)\n"
            "\n"
            "## CRITICAL CONSTRAINTS:\n"
            f"- Target column(s) {self.context.output_targets} are NOT present in sample_df\n"
            f"- Group column {self.context.group_column if self.context.group_column else '(none)'} is NOT present in sample_df (for ranking tasks)\n"
            "- Your pipeline will NEVER see target or group columns during fit or transform\n"
            "- Do NOT reference target or group columns in your pipeline code\n"
            "- Pipeline must work on features-only input\n"
            "\n"
            "## REQUIREMENTS:\n"
            "1. Pipeline MUST accept RAW input features (the columns present in sample_df)\n"
            "2. Pipeline MUST be standalone - do NOT assume data is pre-transformed\n"
            "3. Handle missing values, encode categoricals, scale numerics as needed\n"
            "4. **Feature Scaling**: Strongly recommended for all pipelines\n"
            "   - Neural networks (Keras) require scaled features for gradient descent\n"
            "   - Tree models (XGBoost) don't require scaling but often benefit from it\n"
            "   - Use StandardScaler or MinMaxScaler for numeric columns\n"
            "5. Use FunctionTransformer for custom logic (NOT custom classes)\n"
            "6. Code must define variable named 'pipeline'\n"
            "7. Pipeline output MUST NOT contain object dtypes (use proper encoding)\n"
            "8. Pipeline output MUST NOT contain NaN values (use proper imputation)\n"
            "9. Pipeline MUST NOT change the number of rows (no row filtering)\n"
            "\n"
            "## FORBIDDEN PATTERNS:\n"
            "- NEVER use pd.get_dummies() - it creates inconsistent columns between train/val\n"
            "- Use sklearn.preprocessing.OneHotEncoder instead (learns categories during fit, applies consistently)\n"
            "\n"
            "\n"
            "## TASK:\n"
            "Implement the plan by creating an sklearn Pipeline.\n"
            "\n"
            "## PIPELINE TEMPLATE (examples):\n"
            "```python\n"
            f"{pipeline_template}\n"
            "```\n"
            "\n"
            "## WORKFLOW:\n"
            "1. Review stats_report and task_analysis to understand the data\n"
            "2. Interpret the plan rationale and changes\n"
            "3. Write code that creates the Pipeline\n"
            "4. Ensure 'pipeline' variable is defined\n"
            "5. Call save_pipeline_code(code_string)\n"
            "\n"
            "CRITICAL: Your pipeline will receive RAW data as input. Do not expect pre-engineered features.\n"
        )

        return instructions
