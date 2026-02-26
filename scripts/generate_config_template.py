#!/usr/bin/env python3
"""
Generate config.yaml.template from Pydantic schema.

This script introspects the Config Pydantic model and generates a comprehensive
YAML template showing all available configuration options with:
- Field names
- Types
- Default values
- Descriptions
- Examples

Usage:
    python scripts/generate_config_template.py > config.yaml.template
"""

import sys
from pathlib import Path

# Add plexe to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plexe.config import Config


def format_value(value):
    """Format a value for YAML output."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"[{', '.join(format_value(v) for v in value)}]"
    elif isinstance(value, dict):
        if not value:
            return "{}"
        return str(value)
    else:
        return str(value)


def generate_template():
    """Generate config template from Pydantic schema."""
    schema = Config.model_json_schema()

    print("# " + "=" * 60)
    print("# Plexe Configuration Template (AUTO-GENERATED)")
    print("# " + "=" * 60)
    print("#")
    print("# This file shows all available configuration options.")
    print("# Uncomment and modify values to override defaults.")
    print("# All fields are optional - only specify what you want to change.")
    print("#")
    print("# Generated from: plexe/config.py (Config Pydantic model)")
    print("# To regenerate: python scripts/generate_config_template.py > config.yaml.template")
    print()

    # Group fields by category (based on comments in Config class)
    categories = {
        "Search Settings": [
            "max_search_iterations",
            "max_parallel_variants",
        ],
        "Training Settings": [
            "training_timeout",
            "nn_default_epochs",
            "nn_max_epochs",
            "nn_default_batch_size",
        ],
        "LLM Settings (per agent role)": [
            "statistical_analysis_llm",
            "ml_task_analysis_llm",
            "metric_selection_llm",
            "dataset_splitting_llm",
            "baseline_builder_llm",
            "feature_processor_llm",
            "model_definer_llm",
            "evaluation_llm",
            "hypothesiser_llm",
            "planner_llm",
            "insight_extractor_llm",
        ],
        "Logging & Agent Settings": [
            "log_level",
            "agent_verbosity_level",
        ],
        "OpenTelemetry Tracing": [
            "enable_otel",
            "otel_endpoint",
            "otel_headers",
        ],
        "Evaluation Settings": [
            "performance_threshold",
        ],
        "Sampling Settings": [
            "train_sample_size",
            "val_sample_size",
        ],
        "Code Generation": [
            "allowed_base_imports",
        ],
        "Model Constraints": [
            "allowed_model_types",
        ],
        "Spark Execution": [
            "spark_mode",
            "spark_local_cores",
            "spark_driver_memory",
        ],
        "Databricks Settings": [
            "databricks_use_serverless",
            "databricks_cluster_id",
            "databricks_host",
            "databricks_token",
            "databricks_profile",
        ],
        "Dataset Format": [
            "csv_delimiter",
            "csv_header",
        ],
        "Runtime Overrides": [
            "user_id",
            "experiment_id",
        ],
        "LiteLLM Global Settings": [
            "litellm_ssl_verify",
            "litellm_drop_params",
        ],
        "LiteLLM Routing": [
            "routing_config",
        ],
    }

    properties = schema.get("properties", {})

    for category, fields in categories.items():
        print(f"# {'-' * 60}")
        print(f"# {category}")
        print(f"# {'-' * 60}")
        print()

        for field_name in fields:
            if field_name not in properties:
                continue

            field_info = properties[field_name]
            field_type = field_info.get("type", "unknown")
            description = field_info.get("description", "No description")
            default = field_info.get("default")

            # Special handling for nested routing_config
            if field_name == "routing_config":
                print(f"# {description}")
                print("# Type: RoutingConfig (object)")
                print(f"# Default: {format_value(default)}")
                print("#")
                print("# routing_config:")
                print("#   default:")
                print("#     api_base: null")
                print("#     headers: {}")
                print("#   providers:")
                print("#     my-proxy:")
                print('#       api_base: "https://proxy.example.com/v1"')
                print("#       headers:")
                print('#         authorization: "${PROXY_TOKEN}"')
                print("#   models:")
                print("#     anthropic/claude-sonnet-4-5-20250929: my-proxy")
                print()
                continue

            # Format type (simplify complex types)
            if isinstance(field_type, list):
                field_type = " | ".join(field_type)
            if "anyOf" in field_info:
                field_type = " | ".join(t.get("type", "unknown") for t in field_info["anyOf"])

            # Output field documentation
            print(f"# {description}")
            print(f"# Type: {field_type}")
            print(f"# Default: {format_value(default)}")
            print(f"# {field_name}: {format_value(default)}")
            print()

    print("# " + "=" * 60)
    print("# Example: Minimal Config for Quick Testing")
    print("# " + "=" * 60)
    print("#")
    print("# Uncomment these for quick local testing:")
    print("#")
    print("# agent_verbosity_level: 1")
    print("# max_search_iterations: 5")
    print("# train_sample_size: 10000")
    print("# val_sample_size: 3000")


if __name__ == "__main__":
    generate_template()
