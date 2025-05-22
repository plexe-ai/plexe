"""
This module defines dataclasses for structured model descriptions.

These classes provide a comprehensive representation of a model's
characteristics, including schemas, implementation details, performance
metrics, and source code, organized in a structured format suitable
for various output formats and visualization purposes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class SchemaInfo(DataClassJsonMixin):
    """Information about the model's input and output schemas."""

    input: Dict[str, Any]
    output: Dict[str, Any]


@dataclass
class ImplementationInfo(DataClassJsonMixin):
    """Technical information about the model implementation."""

    framework: Optional[str] = None
    model_type: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    size: Optional[int] = None


@dataclass
class PerformanceInfo(DataClassJsonMixin):
    """Performance metrics and training data information."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CodeInfo(DataClassJsonMixin):
    """Information about the model's source code."""

    training: Optional[str] = None
    prediction: Optional[str] = None
    feature_transformations: Optional[str] = None


@dataclass
class ModelDescription(DataClassJsonMixin):
    """A comprehensive description of a model."""

    id: str
    state: str
    intent: str
    schemas: SchemaInfo
    implementation: ImplementationInfo
    performance: PerformanceInfo
    code: CodeInfo
    training_date: Optional[str] = None
    rationale: Optional[str] = None
    provider: Optional[str] = None
    task_type: Optional[str] = None
    domain: Optional[str] = None
    behavior: Optional[str] = None
    preprocessing_summary: Optional[str] = None
    architecture_summary: Optional[str] = None
    training_procedure: Optional[str] = None
    evaluation_metric: Optional[str] = None
    inference_behavior: Optional[str] = None
    strengths: Optional[str] = None
    limitations: Optional[str] = None

    def as_text(self) -> str:
        """Convert the model description to a formatted text string."""
        # Simple text representation
        lines = [
            f"Model: {self.id}",
            f"State: {self.state}",
            f"Intent: {self.intent}",
            f"Training Date: {self.training_date or 'Not available'}",
            f"Built with: {self.provider or 'Unknown provider'}",
            "",
            "Input Schema:",
            "\n".join(f"  - {k}: {v}" for k, v in self.schemas.input.items()),
            "",
            "Output Schema:",
            "\n".join(f"  - {k}: {v}" for k, v in self.schemas.output.items()),
            "",
            "Implementation:",
            f"  - Framework: {self.implementation.framework or 'Not specified'}",
            f"  - Model Type: {self.implementation.model_type or 'Not specified'}",
            f"  - Size: {self.implementation.size or 'Unknown'} bytes",
            "",
            "Task Information:",
            f"  - Task Type: {self.task_type or 'Not specified'}",
            f"  - Domain: {self.domain or 'Not specified'}",
            f"  - Behavior: {self.behavior or 'Not specified'}",
            "",
            "Technical Details:",
            f"  - Preprocessing: {self.preprocessing_summary or 'Not available'}",
            f"  - Architecture: {self.architecture_summary or 'Not available'}",
            f"  - Training Procedure: {self.training_procedure or 'Not available'}",
            f"  - Evaluation Metric: {self.evaluation_metric or 'Not available'}",
            f"  - Inference Behavior: {self.inference_behavior or 'Not available'}",
            "",
            "Analysis:",
            f"  - Strengths: {self.strengths or 'Not available'}",
            f"  - Limitations: {self.limitations or 'Not available'}",
            "",
            "Performance Metrics:",
            "\n".join(f"  - {k}: {v}" for k, v in self.performance.metrics.items()),
            "",
            "Model Code:",
            "  - Training Code:",
            f"    ```python\n{self.code.training or '# No training code available'}\n```",
            "  - Prediction Code:",
            f"    ```python\n{self.code.prediction or '# No prediction code available'}\n```",
            "  - Feature Transformation Code:",
            f"    ```python\n{self.code.feature_transformations or '# No feature transformation code available'}\n```",
            "",
            "Rationale:",
            self.rationale or "Not available",
        ]
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Convert the model description to a markdown string."""
        # Markdown representation with formatting
        lines = [
            f"# Model: {self.id}",
            "",
            f"**State:** {self.state}",
            f"**Intent:** {self.intent}",
            f"**Training Date:** {self.training_date or 'Not available'}",
            f"**Built with:** {self.provider or 'Unknown provider'}",
            "",
            "## Input Schema",
            "\n".join(f"- `{k}`: {v}" for k, v in self.schemas.input.items()),
            "",
            "## Output Schema",
            "\n".join(f"- `{k}`: {v}" for k, v in self.schemas.output.items()),
            "",
            "## Implementation",
            f"- **Framework:** {self.implementation.framework or 'Not specified'}",
            f"- **Model Type:** {self.implementation.model_type or 'Not specified'}",
            f"- **Size:** {self.implementation.size or 'Unknown'} bytes",
            f"- **Artifacts:** {', '.join(self.implementation.artifacts) or 'None'}",
            "",
            "## Task Information",
            f"- **Task Type:** {self.task_type or 'Not specified'}",
            f"- **Domain:** {self.domain or 'Not specified'}",
            f"- **Behavior:** {self.behavior or 'Not specified'}",
            "",
            "## Technical Details",
            f"- **Preprocessing:** {self.preprocessing_summary or 'Not available'}",
            f"- **Architecture:** {self.architecture_summary or 'Not available'}",
            f"- **Training Procedure:** {self.training_procedure or 'Not available'}",
            f"- **Evaluation Metric:** {self.evaluation_metric or 'Not available'}",
            f"- **Inference Behavior:** {self.inference_behavior or 'Not available'}",
            "",
            "## Analysis",
            f"- **Strengths:** {self.strengths or 'Not available'}",
            f"- **Limitations:** {self.limitations or 'Not available'}",
            "",
            "## Performance Metrics",
            "\n".join(f"- **{k}:** {v}" for k, v in self.performance.metrics.items()),
            "",
            "## Model Code",
            "### Training Code",
            "```python",
            self.code.training or "# No training code available",
            "```",
            "### Prediction Code",
            "```python",
            self.code.prediction or "# No prediction code available",
            "```",
            "### Feature Transformation Code",
            "```python",
            self.code.feature_transformations or "# No feature transformation code available",
            "```",
            "",
            "## Rationale",
            self.rationale or "_Not available_",
        ]
        return "\n".join(lines)
