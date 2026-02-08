"""
Statistical Analyser Agent.

Performs comprehensive statistical analysis on datasets using Spark.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.models import BuildContext
from plexe.config import Config, get_routing_for_model
from plexe.tools.submission import get_register_statistical_profile_tool
from plexe.utils.tracing import agent_span

logger = logging.getLogger(__name__)


class StatisticalAnalyserAgent:
    """
    Agent that performs statistical profiling of datasets.

    Narrow responsibility: Pure statistical analysis (no ML interpretation).
    """

    def __init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config):
        self.spark = spark
        self.dataset_uri = dataset_uri
        self.context = context
        self.config = config
        self.llm_model = config.statistical_analysis_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with structured submission tool."""
        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="StatisticalAnalyser",
            instructions=(
                "## YOUR ROLE:\n"
                "You are a statistical data profiler. Perform comprehensive statistical analysis\n"
                "to characterize data quality, distributions, and patterns. The analysis relates to a broader "
                "ML project, described below.\n"
                "\n"
                "## ML TASK FOR BROADER CONTEXT:\n"
                f'"{self.context.intent}"'
                "\n"
                f"{feedback_section}"
                "## ENVIRONMENT:\n"
                "Your PySpark code has access to the following variables:\n"
                "- `spark`: PySpark SparkSession (use interactively to analyze data)\n"
                "- `dataset_uri`: Path to dataset in parquet format\n"
                "\n"
                "## HOW TO ANALYZE:\n"
                "Write PySpark code directly:\n"
                "  df = spark.read.parquet(dataset_uri)\n"
                "  row_count = df.count()\n"
                "  column_types = df.dtypes\n"
                "  df.show(5, truncate=False)  # View sample rows to understand content\n"
                "  # ... compute statistics appropriate to each column type\n"
                "\n"
                "## REQUIRED ANALYSIS:\n"
                "Analyze all columns, adapting your approach to the data type:\n"
                "1. Dataset shape (total rows, total columns)\n"
                "2. Column types (separate into numeric, categorical, other)\n"
                "3. Missing values (percentage missing per column)\n"
                "4. For numeric columns: mean, std, min, max, quartiles, skewness, kurtosis, outliers\n"
                "5. For categorical columns: cardinality, mode, mode frequency, unique count\n"
                "6. For string columns containing file paths: check validity (e.g., common extensions), sample paths\n"
                "7. For string columns containing text: length statistics (min/max/avg character count)\n"
                "8. Data quality issues (missing values, inconsistencies, suspicious patterns)\n"
                "9. Key statistical insights relevant to the type of dataset\n"
                "\n"
                "## OUTPUT:\n"
                "When analysis is complete, call save_statistical_profile(...) with ALL required parameters.\n"
                "The function signature shows exactly what statistics you must provide.\n"
                "\n"
                "## FOCUS:\n"
                "- Pure statistical analysis ONLY (no ML modeling recommendations)\n"
                "- Be thorough - analyze ALL columns\n"
                "- Adapt your analysis methods to what you see in the data (numbers vs paths vs text)\n"
                "- Identify data quality issues that could affect modeling\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_register_statistical_profile_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["pyspark", "pyspark.*", "pyspark.sql", "pyspark.sql.*", "pyspark.sql.functions"],
            max_steps=20,
        )

    @agent_span("StatisticalAnalyserAgent")
    def run(self) -> dict:
        """Run statistical analysis."""

        logger.info("Starting statistical analysis...")

        agent = self._build_agent()

        agent.run(
            task="Perform comprehensive statistical profiling on the dataset.",
            additional_args={"spark": self.spark, "dataset_uri": self.dataset_uri},
        )

        profile = self.context.scratch.get("_statistical_profile")

        if not profile:
            logger.warning("Agent completed but no profile was saved")
            return {"error": "No profile generated"}

        logger.info("Statistical analysis completed successfully")
        return profile
