"""
Configuration for plexe.

Provides Config Pydantic model, constants, and logging setup.
"""

import importlib.util
import logging
import os
import re
import warnings
import yaml
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

from plexe.models import DataLayout


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*has decorators other than @tool.*", category=UserWarning)

# Suppress cryptography deprecation warnings from smolagents introspection
# (smolagents.local_python_executor introspects modules, triggering cryptography deprecations)
try:
    from cryptography.utils import CryptographyDeprecationWarning

    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass  # cryptography not installed or older version


# ============================================
# Constants
# ============================================


class ModelType:
    """Supported model types (architectural decision)."""

    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    KERAS = "keras"
    PYTORCH = "pytorch"


# Default model types (enabled by default, user can override via --allowed-model-types)
def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


DEFAULT_MODEL_TYPES = [
    ModelType.XGBOOST,
    ModelType.CATBOOST,
    ModelType.LIGHTGBM,
    ModelType.KERAS,
]
if _is_module_available("torch"):
    DEFAULT_MODEL_TYPES.append(ModelType.PYTORCH)

# Task-compatible model types based on data layout
# Maps DataLayout enum to compatible model types
TASK_COMPATIBLE_MODELS = {
    DataLayout.FLAT_NUMERIC: [
        ModelType.XGBOOST,
        ModelType.CATBOOST,
        ModelType.LIGHTGBM,
        ModelType.KERAS,
        ModelType.PYTORCH,
    ],  # Tabular data
    DataLayout.IMAGE_PATH: [ModelType.KERAS, ModelType.PYTORCH],  # Image data
    DataLayout.TEXT_STRING: [ModelType.KERAS, ModelType.PYTORCH],  # Text data
}


class StandardMetric(str, Enum):
    """
    Standard metrics with hardcoded implementations.

    If agent selects a metric not in this enum, MetricImplementationAgent
    will be invoked to generate custom implementation.
    """

    # Classification - Simple
    ACCURACY = "accuracy"

    # Classification - F1 Score variants
    F1_SCORE = "f1_score"
    F1_WEIGHTED = "f1_weighted"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"

    # Classification - Precision variants
    PRECISION = "precision"
    PRECISION_WEIGHTED = "precision_weighted"
    PRECISION_MACRO = "precision_macro"
    PRECISION_MICRO = "precision_micro"

    # Classification - Recall variants
    RECALL = "recall"
    RECALL_WEIGHTED = "recall_weighted"
    RECALL_MACRO = "recall_macro"
    RECALL_MICRO = "recall_micro"

    # Classification - Probabilistic
    ROC_AUC = "roc_auc"
    ROC_AUC_OVR = "roc_auc_ovr"
    ROC_AUC_OVO = "roc_auc_ovo"
    LOG_LOSS = "log_loss"

    # Classification - Other
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    COHEN_KAPPA = "cohen_kappa"
    HAMMING_LOSS = "hamming_loss"

    # Regression
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    MAPE = "mape"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    MAX_ERROR = "max_error"
    EXPLAINED_VARIANCE = "explained_variance"

    # Ranking
    NDCG = "ndcg"
    MAP = "map"
    MRR = "mrr"


# Standard library modules allowed for all agents
STANDARD_LIB_IMPORTS = [
    "pathlib",
    "typing",
    "dataclasses",
    "json",
    "io",
    "time",
    "datetime",
    "os",
    "sys",
    "math",
    "random",
    "itertools",
    "collections",
    "functools",
    "operator",
    "re",
    "copy",
    "warnings",
    "logging",
    "traceback",
]


# ============================================
# Configuration Helpers
# ============================================


def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve ${VAR_NAME} environment variable references.

    Args:
        value: Value to process (can be str, dict, list, or primitive)

    Returns:
        Value with all ${VAR_NAME} references replaced with actual env var values

    Raises:
        ValueError: If referenced environment variable is not set
    """
    if isinstance(value, str):
        # Replace all ${VAR} patterns in the string
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is referenced in config "
                    f"but not set. Please set it before starting."
                )
            return env_value

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    else:
        # Primitives pass through unchanged
        return value


# ============================================
# Custom Settings Source for YAML Config
# ============================================


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads config from YAML file specified by CONFIG_FILE env var."""

    def get_field_value(self, field, field_name):
        """Not used in Pydantic v2 - use __call__ instead."""
        pass

    def __call__(self):
        """Load configuration from YAML file with ${VAR} resolution."""
        config_file = os.getenv("CONFIG_FILE")

        if not config_file:
            return {}

        config_path = Path(config_file)
        if not config_path.exists():
            logging.getLogger(__name__).warning(f"CONFIG_FILE set but file not found: {config_file}")
            return {}

        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f)

            if yaml_data is None:
                return {}

            # Resolve ${VAR} environment variable references
            resolved_data = _resolve_env_vars(yaml_data)

            return resolved_data

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load YAML config from {config_file}: {e}")
            raise


# ============================================
# Configuration Models
# ============================================


class RoutingProviderConfig(BaseModel):
    """Configuration for a single routing provider."""

    api_base: str | None = Field(default=None, description="Base URL for API requests (null = use LiteLLM default)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers for requests")


class RoutingConfig(BaseModel):
    """LiteLLM routing configuration for custom API endpoints."""

    default: RoutingProviderConfig | None = Field(default=None, description="Default routing for all models")
    providers: dict[str, RoutingProviderConfig] = Field(
        default_factory=dict, description="Named provider configurations"
    )
    models: dict[str, str] = Field(default_factory=dict, description="Model ID to provider name mappings")

    @field_validator("models")
    def validate_model_providers(cls, v: dict[str, str], info) -> dict[str, str]:
        """Validate that all model provider references exist."""
        if not v:
            return v

        providers = info.data.get("providers", {})
        for model_id, provider_name in v.items():
            if provider_name not in providers:
                raise ValueError(
                    f"Model '{model_id}' references provider '{provider_name}' "
                    f"which does not exist. Available: {list(providers.keys())}"
                )
        return v


class Config(BaseSettings):
    """Configuration for model building workflow."""

    # Search settings
    max_search_iterations: int = Field(
        default=10, description="Maximum hypothesis rounds in model search (each creates ~3 variants)", gt=0
    )
    max_parallel_variants: int = Field(
        default=3, description="Maximum variants to execute concurrently (controls resource usage)", gt=0
    )

    # Training settings
    training_timeout: int = Field(default=1800, description="Timeout for training runs (seconds)", gt=0)
    nn_default_epochs: int = Field(
        default=25, description="Default epochs for neural network training (Keras, PyTorch)"
    )
    nn_max_epochs: int = Field(default=50, description="Maximum epochs for neural network training (Keras, PyTorch)")
    nn_default_batch_size: int = Field(
        default=32, description="Default batch size for neural network training (Keras, PyTorch)"
    )

    # LLM settings (per agent role)
    statistical_analysis_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for statistical profiling agent"
    )
    ml_task_analysis_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for ML task analysis agent"
    )
    metric_selection_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for metric selection agent"
    )
    dataset_splitting_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for dataset splitting agent"
    )
    baseline_builder_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for baseline builder agent"
    )
    feature_processor_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for feature engineering agent"
    )
    model_definer_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for model definition agent"
    )
    evaluation_llm: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929", description="LLM for model evaluation agent"
    )
    hypothesiser_llm: str = Field(default="openai/gpt-5-mini", description="LLM for hypothesiser agent")
    planner_llm: str = Field(default="openai/gpt-5-mini", description="LLM for planner agent")
    insight_extractor_llm: str = Field(default="openai/gpt-5-mini", description="LLM for insight extractor agent")

    # Logging settings
    log_level: str = Field(default="INFO", description="Python logging level (DEBUG, INFO, WARNING, ERROR)")

    # Agent settings
    agent_verbosity_level: int = Field(
        default=0, description="Smolagents verbosity level (0=silent, 1=normal, 2=verbose)", ge=0, le=2
    )

    # OpenTelemetry tracing settings
    enable_otel: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    otel_endpoint: str | None = Field(
        default=None,
        description="OTLP endpoint URL (e.g., https://cloud.langfuse.com/api/public/otel)",
        validation_alias=AliasChoices("otel_endpoint", "OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    otel_headers: dict[str, str] = Field(default_factory=dict, description="Authentication headers for OTLP endpoint")

    # LiteLLM routing configuration
    routing_config: RoutingConfig | None = Field(default=None, description="Per-model LiteLLM routing configuration")

    # LiteLLM global settings
    litellm_ssl_verify: bool = Field(default=True, description="Enable SSL certificate verification for LiteLLM")
    litellm_drop_params: bool = Field(
        default=False, description="Drop unsupported parameters instead of raising errors"
    )

    # Evaluation settings
    performance_threshold: float = Field(
        default=1.1, description="Minimum improvement over baseline (1.1 = 10% better)"
    )

    # Sampling settings
    train_sample_size: int = Field(default=30_000, description="Training sample size for fast search iterations", gt=0)
    val_sample_size: int = Field(default=10_000, description="Validation sample size for fast search iterations", gt=0)

    # Code generation settings
    allowed_base_imports: list[str] = Field(
        default_factory=lambda: STANDARD_LIB_IMPORTS.copy(),
        description="Standard library modules allowed for agent code generation",
    )

    # Model type constraints
    allowed_model_types: list[str] | None = Field(
        default=None, description="Restrict to specific model types (null = all allowed)"
    )

    # Spark execution settings
    spark_mode: str = Field(default="local", description="Spark backend: 'local' (PySpark) or 'databricks'")

    # Dataset format options
    csv_delimiter: str = Field(default=",", description="CSV delimiter character")
    csv_header: bool = Field(default=True, description="Whether CSV files have header row")

    # Local Spark settings
    spark_local_cores: int = Field(default=8, description="Number of Spark worker threads (local mode only)")
    spark_driver_memory: str = Field(default="8g", description="Spark driver memory (local mode only)")

    # Databricks settings
    databricks_use_serverless: bool = Field(default=False, description="Use Databricks serverless compute")
    databricks_cluster_id: str | None = Field(default=None, description="Databricks cluster ID (if specified)")
    databricks_host: str | None = Field(default=None, description="Databricks workspace URL")
    databricks_token: str | None = Field(default=None, description="Databricks access token")
    databricks_profile: str | None = Field(
        default=None,
        description="Databricks config profile name",
        validation_alias=AliasChoices("databricks_profile", "DATABRICKS_CONFIG_PROFILE"),
    )

    # Runtime overrides
    user_id: str | None = Field(default=None, description="User identifier (set at runtime)")
    experiment_id: str | None = Field(default=None, description="Experiment identifier (set at runtime)")

    model_config = SettingsConfigDict(
        extra="ignore",  # Ignore unknown fields from YAML
        validate_assignment=True,  # Validate when fields are modified
        case_sensitive=False,  # Case-insensitive env var matching (USER_ID → user_id)
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Customize settings source priority.

        Priority (highest to lowest):
        1. init_settings: Explicit constructor arguments (CLI overrides)
        2. env_settings: Environment variables
        3. YamlConfigSettingsSource: YAML file (with ${VAR} resolution)
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="after")
    def validate_nn_training_settings(self) -> "Config":
        """Ensure neural network defaults do not exceed the configured cap."""
        if self.nn_default_epochs > self.nn_max_epochs:
            raise ValueError("nn_default_epochs must be <= nn_max_epochs")
        return self

    @model_validator(mode="after")
    def parse_otel_headers_from_env(self) -> "Config":
        """Parse OTEL_EXPORTER_OTLP_HEADERS (comma-separated key=value pairs)."""
        if os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
            # Parse comma-separated key=value pairs (standard OTEL format)
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            headers_dict = dict(self.otel_headers)  # Copy existing
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers_dict[key.strip()] = value.strip()
                else:
                    logging.getLogger(__name__).warning(
                        f"Skipping malformed header pair in OTEL_EXPORTER_OTLP_HEADERS: '{pair}' (expected KEY=VALUE)"
                    )
            object.__setattr__(self, "otel_headers", headers_dict)

        return self


# ============================================
# Routing Helpers
# ============================================


def get_routing_for_model(config: RoutingConfig | None, model_id: str) -> tuple[str | None, dict[str, str]]:
    """
    Get routing configuration for a specific model ID.

    Lookup order:
    1. Check if model_id is in 'models' mapping → use that provider's config
    2. Else use 'default' config if present
    3. Else return (None, {}) for LiteLLM's default routing

    Args:
        config: Routing configuration (or None if no config loaded)
        model_id: Model ID to look up (e.g., "anthropic/claude-sonnet-4-5-20250929")

    Returns:
        Tuple of (api_base, headers) where:
        - api_base: Base URL for API requests (None = use LiteLLM default)
        - headers: Dict of HTTP headers to include in requests
    """
    # If no config provided, use LiteLLM defaults
    if config is None:
        return None, {}

    # Check if model has explicit provider mapping
    if model_id in config.models:
        provider_name = config.models[model_id]

        if provider_name not in config.providers:
            # This should have been caught by validation, but handle gracefully
            logging.getLogger(__name__).warning(
                f"Model '{model_id}' references non-existent provider '{provider_name}'. Using default routing."
            )
            provider_config = config.default
        else:
            provider_config = config.providers[provider_name]
            logging.getLogger(__name__).debug(f"Model '{model_id}' → provider '{provider_name}'")
    else:
        # No explicit mapping, use default
        provider_config = config.default
        logging.getLogger(__name__).debug(f"Model '{model_id}' → default routing")

    # If no applicable config found, use LiteLLM defaults
    if provider_config is None:
        return None, {}

    return provider_config.api_base, provider_config.headers


# ============================================
# Logging Setup
# ============================================


def setup_logging(config: Config) -> logging.Logger:
    """
    Configure logging for the plexe package.

    Args:
        config: Configuration object

    Returns:
        Configured package logger
    """
    # Get package root logger
    package_logger = logging.getLogger("plexe")
    package_logger.setLevel(getattr(logging, config.log_level.upper()))

    # Clear existing handlers to avoid duplicates
    package_logger.handlers = []

    # Define color-coded formatter
    class ColoredFormatter(logging.Formatter):
        """Formatter that adds colors to log levels."""

        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"

        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            return super().format(record)

    formatter = ColoredFormatter("[%(asctime)s - %(levelname)s - %(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    package_logger.addHandler(console_handler)

    package_logger.info(f"Logging configured at {config.log_level} level")

    if config.enable_otel:
        package_logger.info("OpenTelemetry tracing enabled")

    return package_logger


def setup_litellm(config: Config) -> None:
    """
    Configure LiteLLM global settings.

    Args:
        config: Configuration object with LiteLLM settings
    """
    try:
        import litellm

        # Apply global settings
        litellm.ssl_verify = config.litellm_ssl_verify
        litellm.drop_params = config.litellm_drop_params

        logger = logging.getLogger(__name__)
        logger.info(
            f"LiteLLM configured: ssl_verify={config.litellm_ssl_verify}, " f"drop_params={config.litellm_drop_params}"
        )
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("litellm not installed, skipping LiteLLM configuration")


# ============================================
# Environment Helpers
# ============================================


def get_config() -> Config:
    """
    Get configuration from YAML file (if specified) with environment variable overrides.

    Loading order (via custom settings sources):
    1. Explicit constructor args (CLI overrides) - highest priority
    2. Environment variables (auto-detected by BaseSettings)
    3. YAML file (if CONFIG_FILE env var set, with ${VAR} resolution)
    4. Field defaults - lowest priority

    Note: Environment variables override YAML values when both are present.

    Returns:
        Config instance with all overrides applied

    Raises:
        FileNotFoundError: If CONFIG_FILE is set but file doesn't exist
        ValueError: If config is invalid
    """
    # Config() constructor automatically uses custom settings sources
    # Priority: constructor args > env vars > YAML file > defaults
    return Config()
