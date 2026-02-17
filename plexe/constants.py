"""
Constants for plexe.

Centralizes magic strings and configuration values.
"""

# ============================================
# Context Scratch Keys
# ============================================


class ScratchKeys:
    """
    Keys for BuildContext.scratch dictionary.

    Centralizes all magic strings used for agent communication.
    """

    # Agent outputs
    SAVED_PIPELINE = "_saved_pipeline"
    SAVED_MODEL = "_saved_model"
    STATISTICAL_PROFILE = "_statistical_profile"
    EDA_REPORT = "_eda_report"

    # Dataset URIs
    TRAIN_URI = "_train_uri"
    VAL_URI = "_val_uri"
    TEST_URI = "_test_uri"
    TRAIN_SAMPLE_URI = "_train_sample_uri"
    VAL_SAMPLE_URI = "_val_sample_uri"

    # Baseline info
    BASELINE_NAME = "_baseline_name"
    BASELINE_DESCRIPTION = "_baseline_description"


# ============================================
# Dataset Naming Patterns
# ============================================


class DatasetPatterns:
    """Naming patterns for dataset artifacts."""

    TRAIN_SPLIT = "train.parquet"
    VAL_SPLIT = "val.parquet"
    TEST_SPLIT = "test.parquet"

    TRAIN_SAMPLE = "train_sample.parquet"
    VAL_SAMPLE = "val_sample.parquet"

    @staticmethod
    def transformed_name(base_uri: str, iteration: int) -> str:
        """Generate name for transformed dataset."""
        return f"{base_uri}_iter{iteration}_transformed"


# ============================================
# Search Constants
# ============================================


class SearchDefaults:
    """Default values for search configuration."""

    NUM_DRAFTS = 3  # Number of diverse initial solutions
    DEBUG_PROB = 0.3  # Probability of debugging vs improving
    MAX_DEBUG_DEPTH = 2  # Stop debugging after N consecutive debug steps
    STAGNATION_WINDOW = 8  # Check for stagnation over last N iterations
    STAGNATION_THRESHOLD = 0.001  # Improvement threshold to avoid early stop
    IMPROVEMENT_THRESHOLD = 0.10  # 10% improvement over baseline triggers stop


# ============================================
# Directory and File Names
# ============================================


class DirNames:
    """Standard directory and file names used across the codebase."""

    BUILD_DIR = ".build"  # Working directory for intermediate artifacts


# ============================================
# Workflow Phase Names
# ============================================


class PhaseNames:
    """
    Standardized phase names for workflow orchestration.

    Used for:
    - Checkpoint filenames
    - Pause points
    - Resume logic
    """

    ANALYZE_DATA = "01_analyze_data"
    PREPARE_DATA = "02_prepare_data"
    BUILD_BASELINES = "03_build_baselines"
    SEARCH_MODELS = "04_search_models"
    EVALUATE_FINAL = "05_evaluate_final"
    PACKAGE_FINAL_MODEL = "06_package_final_model"
