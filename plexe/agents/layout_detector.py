"""
Layout Detection Agent.

Determines the physical structure of the dataset (data layout) and identifies
the primary input column for non-tabular data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.models import BuildContext
from plexe.config import Config
from plexe.tools.submission import get_register_layout_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class LayoutDetectionAgent:
    """
    Agent that detects data layout and identifies primary input column.

    Narrow responsibility: Determine what kind of data structure we're dealing with.
    """

    def __init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config):
        self.spark = spark
        self.dataset_uri = dataset_uri
        self.context = context
        self.config = config
        self.llm_model = config.statistical_analysis_llm  # Reuse same LLM as stats

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with layout detection tool."""
        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="LayoutDetector",
            instructions=(
                "## YOUR ROLE:\n"
                "You are a data structure analyst. Your job is to determine what kind of data\n"
                "a dataset contains based on its schema, sample data, and the broader ML project in the context "
                "of which it is being inspected.\n"
                "\n"
                "## ML PROJECT (FOR CONTEXT):\n"
                f'"{self.context.intent}"'
                "\n"
                "## DATA LAYOUTS:\n"
                "There are four possible data layouts:\n"
                "\n"
                "1. **FLAT_NUMERIC**: Traditional tabular data\n"
                "   - Contains numeric columns (int, float, double) and/or categorical columns (string)\n"
                "   - Used for classic ML tasks like predicting customer churn, house prices, etc.\n"
                "   - Example: A CSV with columns like [age, income, city, purchased]\n"
                "   - Primary input column: N/A (all non-target columns are features)\n"
                "\n"
                "2. **IMAGE_PATH**: Image datasets\n"
                "   - Contains a SINGLE string column with file paths to images\n"
                "   - Used for image classification, object detection, etc.\n"
                "   - Example: A parquet with columns like [image_path, label] where image_path='/data/img001.jpg'\n"
                "   - Primary input column: The column containing image file paths\n"
                "\n"
                "3. **TEXT_STRING**: Text document datasets\n"
                "   - Contains a SINGLE string column with text documents\n"
                "   - Used for NLP tasks like sentiment analysis, document classification, etc.\n"
                "   - Example: A parquet with columns like [review_text, rating] where review_text='This product is great!'\n"
                "   - Primary input column: The column containing text content\n"
                "\n"
                "4. **UNSUPPORTED**: Data structure not supported in v1\n"
                "   - Use this when the data doesn't fit any of the above layouts\n"
                "   - Examples requiring UNSUPPORTED:\n"
                "     * Video file paths (.mp4, .avi, .mov)\n"
                "     * Audio file paths (.mp3, .wav)\n"
                "     * Multiple image/text columns (e.g., [image1, image2, label] or [title, body, label])\n"
                "     * Binary blob columns (inline binary data instead of paths)\n"
                "     * Time series data without clear structure\n"
                "     * Mixed/hybrid layouts (e.g., images + text together)\n"
                "   - When using UNSUPPORTED, you MUST provide a clear and concise explanation why\n"
                "\n"
                "## ENVIRONMENT:\n"
                "Your PySpark code has access to these predefined variables:\n"
                "- `spark`: PySpark SparkSession\n"
                "- `dataset_uri`: Path to dataset in parquet format\n"
                "\n"
                "## HOW TO DETECT:\n"
                "Write PySpark code to inspect the schema and sample data:\n"
                "  df = spark.read.parquet(dataset_uri)\n"
                "  df.printSchema()  # View column types\n"
                "  df.show(5, truncate=False)  # View sample rows\n"
                "  # Inspect specific columns to understand content\n"
                "\n"
                "## DECISION LOGIC:\n"
                "1. Look at column types (numeric, string, etc.)\n"
                "2. Sample a few rows to see actual content\n"
                "3. Match content against the user's intent\n"
                "4. Identify which column contains the primary input data (ONLY for IMAGE_PATH or TEXT_STRING)\n"
                "\n"
                "**Disambiguation hints**:\n"
                "- If mostly numeric/categorical columns → likely FLAT_NUMERIC\n"
                "- String columns with short values ('/path/to/image.jpg') → possibly IMAGE_PATH\n"
                "- String columns with long values ('This is a review...') → possibly TEXT_STRING\n"
                "\n"
                "## OUTPUT:\n"
                "When detection is complete, call register_layout(data_layout, primary_input_column, reason).\n"
                "- data_layout: One of 'flat_numeric', 'image_path', 'text_string', 'unsupported'\n"
                "- primary_input_column: Column name (required for IMAGE_PATH/TEXT_STRING, None for FLAT_NUMERIC/UNSUPPORTED)\n"
                "- reason: Required explanation if data_layout='unsupported', otherwise None\n"
                "\n"
                "## FOCUS:\n"
                "- Inspect the data carefully\n"
                "- Use the user's intent to disambiguate\n"
                "- Be confident in your decision\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_register_layout_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["pyspark", "pyspark.*", "pyspark.sql", "pyspark.sql.*", "pyspark.sql.functions"],
            max_steps=10,  # Should be quick - just schema inspection
        )

    @agent_span("LayoutDetectionAgent")
    def run(self) -> dict:
        """Run layout detection."""

        logger.info("Starting layout detection...")

        agent = self._build_agent()

        agent.run(
            task="Detect the data layout and identify the primary input column (if applicable).",
            additional_args={"spark": self.spark, "dataset_uri": self.dataset_uri},
        )

        layout_info = self.context.scratch.get("_layout_info")

        if not layout_info:
            logger.warning("Agent completed but no layout was detected")
            return {"error": "No layout detected"}

        logger.info(f"Layout detection completed: {layout_info['data_layout']}")
        return layout_info
