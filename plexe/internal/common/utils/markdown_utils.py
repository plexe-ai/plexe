"""Utilities for markdown formatting of reports and data."""

from typing import Dict, List, Any, Union


def format_eda_report_markdown(eda_report: Dict[Any, Any]) -> str:
    """
    Convert an EDA report dictionary to a well-formatted markdown document.

    Args:
        eda_report: Dictionary containing the EDA report data

    Returns:
        Formatted markdown string
    """
    if not eda_report:
        return ""

    markdown = [f"# Exploratory Data Analysis Report: {eda_report.get('dataset_name', 'Dataset')}\n"]
    markdown.append(f"Generated: {eda_report.get('timestamp', '')}\n")

    # Dataset Overview
    if overview := eda_report.get("overview", {}):
        markdown.append("## Dataset Overview\n")
        for key, value in overview.items():
            if isinstance(value, dict):
                markdown.append(f"### {str(key).replace('_', ' ').title()}\n")
                markdown.append(_dict_to_markdown(value, level=3))
            else:
                markdown.append(f"**{str(key).replace('_', ' ').title()}**: {value}\n")
        markdown.append("\n")

    # Feature Analysis
    if feature_analysis := eda_report.get("feature_analysis", {}):
        markdown.append("## Feature Analysis\n")
        markdown.append(_dict_to_markdown(feature_analysis, level=2))
        markdown.append("\n")

    # Relationships
    if relationships := eda_report.get("relationships", {}):
        markdown.append("## Feature Relationships\n")
        markdown.append(_dict_to_markdown(relationships, level=2))
        markdown.append("\n")

    # Data Quality
    if data_quality := eda_report.get("data_quality", {}):
        markdown.append("## Data Quality Issues\n")
        markdown.append(_dict_to_markdown(data_quality, level=2))
        markdown.append("\n")

    # Key Insights
    if insights := eda_report.get("insights", []):
        markdown.append("## Key Insights\n")
        for insight in insights:
            markdown.append(f"- {insight}\n")
        markdown.append("\n")

    # Recommendations
    if recommendations := eda_report.get("recommendations", []):
        markdown.append("## Modeling Recommendations\n")
        for recommendation in recommendations:
            markdown.append(f"- {recommendation}\n")

    return "\n".join(markdown)


def _dict_to_markdown(data: Union[Dict, List, Any], level: int = 0) -> str:
    """
    Helper function to convert nested dictionaries and lists to markdown.

    Args:
        data: Data to convert (dict, list, or scalar value)
        level: Header level for section titles

    Returns:
        Markdown string representation
    """
    if not data:
        return ""

    result = []

    if isinstance(data, dict):
        for key, value in data.items():
            header_prefix = "#" * (level + 1)
            key_formatted = str(key).replace("_", " ").title()

            if isinstance(value, dict):
                result.append(f"{header_prefix} {key_formatted}\n")
                result.append(_dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                result.append(f"{header_prefix} {key_formatted}\n")
                for item in value:
                    if isinstance(item, dict):
                        result.append(_dict_to_markdown(item, level + 1))
                    else:
                        result.append(f"- {item}\n")
            else:
                result.append(f"**{key_formatted}**: {value}\n")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                result.append(_dict_to_markdown(item, level))
            else:
                result.append(f"- {item}\n")
    else:
        # Handle scalar value
        result.append(str(data))

    return "\n".join(result)
