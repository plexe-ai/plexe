"""
Dataset Splitter Agent.

Generates and executes PySpark code for intelligent dataset splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.models import BuildContext
from plexe.config import Config
from plexe.tools.submission import get_save_split_uris_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model
from plexe.validation.validators import validate_dataset_splits

logger = logging.getLogger(__name__)


class DatasetSplitterAgent:
    """
    Agent that generates PySpark code for intelligent dataset splitting.

    Handles:
    - Stratified splits for classification (preserves class balance)
    - Chronological splits for time-series (no temporal leakage)
    - Group-aware splits (keeps related records together)
    - Standard random splits (fallback)
    """

    def __init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config):
        """
        Initialize agent.

        Args:
            spark: SparkSession for data access
            dataset_uri: URI to full dataset
            context: Build context with task analysis
            config: Configuration
        """
        self.spark = spark
        self.dataset_uri = dataset_uri
        self.context = context
        self.config = config
        self.llm_model = config.dataset_splitting_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with splitting tool."""
        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="DatasetSplitter",
            instructions=(
                "## YOUR ROLE:\n"
                "Intelligently split datasets into train/validation/test sets. This is NOT trivial - "
                "how you split SIGNIFICANTLY impacts model quality, validity, and data leakage prevention.\n"
                "\n"
                "## CONTEXT AVAILABLE:\n"
                "Review prior analysis to inform your strategy:\n"
                "- `stats_report`: Column types, distributions, missing values, total rows\n"
                "- `task_analysis`: Task type, target columns, data challenges, class balance info\n"
                "\n"
                "## CODE ENVIRONMENT:\n"
                "Your PySpark code has access to these variables:\n"
                "- `spark`: SparkSession\n"
                "- `dataset_uri`: Path to parquet dataset\n"
                "- `split_ratios`: Dict with 'train'/'val'/'test' fractions (e.g., {\"train\": 0.7, \"val\": 0.15, \"test\": 0.15})\n"
                "- `output_dir`: Directory for writing split files\n"
                "- `task_type`: Task type string (e.g., 'binary_classification', 'regression', 'time_series')\n"
                "- `output_targets`: List of target column names\n"
                "\n"
                "## STRATEGY SELECTION:\n"
                "\n"
                "Choose based on data characteristics:\n"
                "1. **Time Series**: If temporal columns exist AND task requires forecasting → Chronological split (prevent future leakage)\n"
                "2. **Classification**: If task_type contains 'classification' → Stratified split (preserve class balance)\n"
                "3. **Small Datasets**: If <10,000 rows → Use 90/5/5 instead of given ratios\n"
                "4. **Group Preservation**: If user_id, session_id, group_id columns exist → Keep groups intact\n"
                "5. **Regression** (default): Random split\n"
                "\n"
                "## TEMPORAL SPLIT DECISION GUIDE:\n"
                "\n"
                "IMPORTANT: Timestamp columns do NOT automatically require chronological splitting.\n"
                "\n"
                "**USE chronological split when:**\n"
                "- Forecasting future values (predict tomorrow from today)\n"
                "- Target has trends/seasonality that change over time\n"
                "- Concept drift likely (patterns evolve)\n"
                "- Production will predict on FUTURE time periods\n"
                "\n"
                "**Random/stratified split is acceptable when:**\n"
                "- Timestamp is metadata only (created_at, upload_date)\n"
                "- Cross-sectional classification (predict attribute, not future event)\n"
                "- Target distribution stable across time\n"
                "- Time is a feature, not the prediction axis\n"
                "\n"
                "**Quick test:** Will model predict SAME time period (unseen records) → random OK. "
                "Will model predict FUTURE time periods → chronological required.\n"
                "\n"
                "## KEY PATTERNS:\n"
                "\n"
                "**Stratified (Classification):**\n"
                "Use `df.sampleBy(target_col, fractions, seed=42)` to sample proportionally from each class.\n"
                "Apply to train, then remainder for val/test.\n"
                "\n"
                "**Chronological (Time-Series):**\n"
                "Detect time column → sort by it → use Window.row_number() → filter by cutoffs.\n"
                "Train=oldest, Test=newest to simulate production (no future leakage).\n"
                "\n"
                "**Random (Regression):**\n"
                "Use `df.randomSplit(weights, seed=42)` with normalized ratios.\n"
                "\n"
                "## YOUR TASK:\n"
                "1. Load: df = spark.read.parquet(dataset_uri)\n"
                "2. Inspect stats_report/task_analysis for time columns, groups, imbalance\n"
                "3. Generate appropriate PySpark split code\n"
                "4. Write to {output_dir}/train.parquet, val.parquet, test.parquet (mode='overwrite')\n"
                "5. Call save_split_uris(train_uri, val_uri, test_uri)\n"
                "\n"
                "## CRITICAL RULES:\n"
                "- PySpark only (NOT pandas) - data may be 200GB\n"
                "- seed=42 for reproducibility\n"
                "- Classification → MUST stratify (unless forecasting task)\n"
                "- Forecasting/predicting future → MUST be chronological\n"
                "- Timestamp alone does NOT mandate chronological split (see TEMPORAL SPLIT DECISION GUIDE)\n"
                "- NO data leakage between splits\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                temperature=self.config.get_temperature("dataset_splitter"),
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_split_uris_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + [
                "posixpath",
                "pyspark",
                "pyspark.*",
                "pyspark.sql",
                "pyspark.sql.*",
                "pyspark.sql.functions",
                "pyspark.sql.window",
            ],
            max_steps=20,
        )

    @agent_span("DatasetSplitterAgent")
    def run(self, split_ratios: dict[str, float], output_dir: str | Path) -> tuple[str, str, str]:
        """
        Generate and execute intelligent dataset splitting.

        Args:
            split_ratios: Split ratios (train/val/test)
            output_dir: Output directory for splits (local Path or S3 URI string)

        Returns:
            (train_uri, val_uri, test_uri)
        """

        logger.info(f"Starting intelligent dataset splitting with ratios: {split_ratios}")

        # Convert to Path if local path string, or keep as S3 URI string
        if isinstance(output_dir, str):
            if output_dir.startswith("s3://"):
                # S3 URI - skip mkdir, Spark will create on write
                output_dir_str = output_dir
                logger.info(f"Output to S3: {output_dir}")
            else:
                # Local path string - convert to Path and create directory
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir_str = str(output_dir)
        else:
            # Already a Path - create directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_str = str(output_dir)

        # Build agent
        agent = self._build_agent()

        # Build task prompt (use string version of output_dir)
        task = self._build_task_prompt(split_ratios, output_dir_str)

        # Run agent
        try:
            agent.run(
                task=task,
                additional_args={
                    "spark": self.spark,
                    "dataset_uri": self.dataset_uri,
                    "split_ratios": split_ratios,
                    "output_dir": output_dir_str,  # Always string now
                    "task_type": self.context.task_analysis.get("task_type", "unknown"),
                    "output_targets": self.context.output_targets,
                    "stats_report": self.context.stats,
                    "task_analysis": self.context.task_analysis,
                },
            )

            # Retrieve URIs from scratch
            train_uri = self.context.scratch.get("_train_uri")
            val_uri = self.context.scratch.get("_val_uri")
            test_uri = self.context.scratch.get("_test_uri")  # Can be None for 2-way splits

            if not train_uri or not val_uri:
                raise ValueError("Agent did not save required split URIs (train and val)")

            # Validate splits exist
            is_valid, error = validate_dataset_splits(self.spark, train_uri, val_uri, test_uri, split_ratios)
            if not is_valid:
                raise ValueError(error)

            if test_uri:
                logger.info(f"Dataset splitting complete: train={train_uri}, val={val_uri}, test={test_uri}")
            else:
                logger.info(f"Dataset splitting complete: train={train_uri}, val={val_uri} (no test set)")

            return train_uri, val_uri, test_uri

        except Exception as e:
            logger.error(f"Dataset splitting failed: {e}")
            raise

    def _build_task_prompt(self, split_ratios: dict[str, float], output_dir: str) -> str:
        """Build task prompt for agent."""

        task_type = self.context.task_analysis.get("task_type", "unknown")
        output_targets = self.context.output_targets
        data_challenges = self.context.task_analysis.get("data_challenges", [])
        recommended_split = self.context.task_analysis.get("recommended_split", {})

        prompt = (
            f"Split the dataset into train/validation/test sets.\n"
            f"\n"
            f"Task Type: {task_type}\n"
            f"Output Targets: {output_targets}\n"
            f"Split Ratios: {split_ratios}\n"
            f"Output Directory: {output_dir}\n"
            f"Data Challenges: {data_challenges}\n"
        )

        # Surface the recommended split strategy from MLTaskAnalyser if available
        if recommended_split:
            prompt += "\n## RECOMMENDED SPLIT STRATEGY (from prior analysis):\n"
            if recommended_split.get("temporal_reasoning"):
                prompt += f"- Temporal: {recommended_split.get('temporal_reasoning')}\n"
            if recommended_split.get("stratification_reasoning"):
                prompt += f"- Stratification: {recommended_split.get('stratification_reasoning')}\n"
            prompt += "\nFollow this recommendation. You determine the appropriate columns.\n"

        prompt += (
            "\n"
            "Based on the task type and data characteristics, choose the appropriate splitting strategy:\n"
            "- Classification → Stratified split (preserve class balance)\n"
            "- Forecasting future events/values → Chronological split (train on past, test on future)\n"
            "- Timestamp exists but task is cross-sectional → Random/stratified split is acceptable\n"
            "- Small dataset (<10K rows) → Adjust ratios to 90/5/5\n"
            "- Group-based data (user_id, session_id) → Preserve groups within splits\n"
            "- Regression → Simple random split\n"
            "\n"
            "Review the stats_report and task_analysis (available in additional_args) to detect:\n"
            "- Temporal columns (datetime types or time/date in name)\n"
            "- Group columns (user_id, customer_id, session_id patterns)\n"
            "- Class imbalance (from target_analysis)\n"
            "- Dataset size (from stats_report)\n"
            "\n"
            "Generate PySpark code to:\n"
            "1. Load dataset from dataset_uri\n"
            "2. Apply appropriate split strategy (use examples in instructions)\n"
            "3. Write three parquet files to output_dir:\n"
            "   - {output_dir}/train.parquet\n"
            "   - {output_dir}/val.parquet\n"
            "   - {output_dir}/test.parquet\n"
            "4. Call save_split_uris(train_path, val_path, test_path) with the full file paths\n"
            "\n"
            "CRITICAL: Ensure splits are appropriate for this specific dataset and task.\n"
        )

        return prompt
