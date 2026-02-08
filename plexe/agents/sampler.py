"""
Sampling Agent.

Generates PySpark code for intelligent dataset sampling.
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
from plexe.tools.submission import get_save_sample_uris_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class SamplingAgent:
    """
    Agent that generates PySpark code for intelligent dataset sampling.

    Handles:
    - Stratified sampling for classification (preserves class balance)
    - Random sampling for regression
    - Representative sampling that maintains data characteristics
    """

    def __init__(self, spark: SparkSession, context: BuildContext, config: Config):
        """
        Initialize agent.

        Args:
            spark: SparkSession for data access
            context: Build context with task analysis
            config: Configuration
        """
        self.spark = spark
        self.context = context
        self.config = config
        self.llm_model = config.dataset_splitting_llm  # Reuse same model

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with sampling tool."""
        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="Sampler",
            instructions=(
                "## YOUR ROLE:\n"
                "Generate PySpark code to create representative samples from train and validation datasets.\n"
                "Create BOTH samples in one run using consistent sampling logic.\n"
                "\n"
                "## CODE ENVIRONMENT:\n"
                "Your PySpark code has access to:\n"
                "- `spark`: SparkSession\n"
                "- `train_uri`: Path to training dataset (parquet)\n"
                "- `val_uri`: Path to validation dataset (parquet)\n"
                "- `train_sample_size`: Target rows for train sample\n"
                "- `val_sample_size`: Target rows for val sample\n"
                "- `output_dir`: Directory for writing samples\n"
                "- `task_type`: Task type (e.g., 'binary_classification', 'regression')\n"
                "- `output_targets`: List of target column names\n"
                "\n"
                "## SAMPLING STRATEGIES:\n"
                "\n"
                "**Classification (Stratified):**\n"
                "Preserve class balance in both samples:\n"
                "- Calculate class fractions based on target sample sizes\n"
                "- Use `df.sampleBy(target_col, fractions, seed=42)`\n"
                "- Apply SAME strategy to both train and val\n"
                "- Make sure all classes are represented, unless EXTREMELY rare\n"  # FIXME: methodologically problematic
                "\n"
                "**Regression (Random):**\n"
                "Simple random sampling:\n"
                "- Calculate fraction: sample_size / total_rows\n"
                "- Use `df.sample(fraction=frac, seed=42)`\n"
                "\n"
                "## ⚠️ CRITICAL SAMPLING REQUIREMENTS:\n"
                "\n"
                "1. **Categorical Coverage (PREVENTS ENCODER ERRORS):**\n"
                "   - For categorical/string columns (non-target), ensure ALL unique values in sample\n"
                "   - Check distinct count: if <1000 categories, include all in sample\n"
                "   - Use stratified sampling or union approach to guarantee coverage\n"
                "   - Why: OneHotEncoder/LabelEncoder trained on sample must see all categories\n"
                "         that appear in full dataset, otherwise transform fails\n"
                "\n"
                "2. **Missing Value Patterns (PREVENTS IMPUTER ERRORS):**\n"
                "   - If full dataset has missing values in any column, sample MUST include some nulls\n"
                "   - Check: df.select([count(when(col(c).isNull(), 1)) for c in columns])\n"
                "   - Why: Imputers need to see null values to learn proper strategy\n"
                "         Sample with no nulls → imputer not fitted → full transform breaks\n"
                "\n"
                "3. **Target Class Coverage (PREVENTS TRAINING ERRORS):**\n"
                "   - For classification: ALL target classes must appear in sample\n"
                "   - For rare classes (<1%), oversample to ensure representation\n"
                "   - Minimum: 10 samples per class if dataset allows\n"
                "   - Why: Model training requires all classes, missing class breaks fit()\n"
                "\n"
                "## YOUR TASK:\n"
                "1. Load both datasets from train_uri and val_uri\n"
                "2. Choose sampling strategy based on task_type\n"
                "3. Create train_sample (using train_sample_size)\n"
                "4. Create val_sample (using val_sample_size)\n"
                "5. Write to:\n"
                "   - {output_dir}/train_sample.parquet\n"
                "   - {output_dir}/val_sample.parquet\n"
                "6. Call save_sample_uris(train_sample_uri, val_sample_uri)\n"
                "\n"
                "## RULES:\n"
                "- Use PySpark (NOT pandas)\n"
                "- Classification: MUST stratify both samples\n"
                "- seed=42 for reproducibility\n"
                "- Apply SAME strategy to both samples\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_save_sample_uris_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["posixpath", "pyspark", "pyspark.*", "pyspark.sql", "pyspark.sql.*", "pyspark.sql.functions"],
            max_steps=15,
        )

    @agent_span("SamplingAgent")
    def run(
        self, train_uri: str, val_uri: str, train_sample_size: int, val_sample_size: int, output_dir: Path
    ) -> tuple[str, str]:
        """
        Generate and execute intelligent sampling for both train and val datasets.

        Args:
            train_uri: URI to training dataset
            val_uri: URI to validation dataset
            train_sample_size: Target rows for train sample
            val_sample_size: Target rows for val sample
            output_dir: Directory for writing samples

        Returns:
            (train_sample_uri, val_sample_uri): Paths to sampled datasets
        """

        logger.info(f"Creating samples: train={train_sample_size} rows, val={val_sample_size} rows")

        # Create output directory if local (S3 paths are created by Spark automatically)
        if not str(output_dir).startswith("s3://"):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        agent = self._build_agent()

        task_type = self.context.task_analysis.get("task_type", "unknown")
        output_targets = self.context.output_targets

        task = (
            f"Create representative samples from train and validation datasets.\n"
            f"\n"
            f"Train Dataset: {train_uri} → {train_sample_size} rows\n"
            f"Val Dataset: {val_uri} → {val_sample_size} rows\n"
            f"Output Directory: {output_dir}\n"
            f"Task Type: {task_type}\n"
            f"Output Targets: {output_targets}\n"
            f"\n"
            f"Use {'stratified' if 'classification' in task_type else 'random'} sampling.\n"
            f"Apply the SAME strategy to both datasets for consistency.\n"
        )

        agent.run(
            task=task,
            additional_args={
                "spark": self.spark,
                "train_uri": train_uri,
                "val_uri": val_uri,
                "train_sample_size": train_sample_size,
                "val_sample_size": val_sample_size,
                "output_dir": str(output_dir),
                "task_type": task_type,
                "output_targets": output_targets,
            },
        )

        # Retrieve URIs from scratch
        train_sample_uri = self.context.scratch.get("_train_sample_uri")
        val_sample_uri = self.context.scratch.get("_val_sample_uri")

        if not all([train_sample_uri, val_sample_uri]):
            raise ValueError("Agent did not save both sample URIs")

        logger.info(f"Samples created: train={train_sample_uri}, val={val_sample_uri}")

        return train_sample_uri, val_sample_uri
