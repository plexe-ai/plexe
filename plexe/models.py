"""
Simple dataclasses for model building workflow.

No heavy abstractions - just data containers.
"""

import inspect
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Literal

from sklearn.pipeline import Pipeline


# ============================================
# Data Layout (i.e. what 'structure' the input data has
# ============================================


class DataLayout(str, Enum):
    """
    Physical structure of dataset (not semantic meaning).

    Determines preprocessing approach and model architecture selection.
    """

    FLAT_NUMERIC = "flat_numeric"  # Traditional tabular data in parquet
    IMAGE_PATH = "image_path"  # Image file paths stored in parquet
    TEXT_STRING = "text_string"  # Text documents as strings in parquet
    UNSUPPORTED = "unsupported"  # Data structure not supported (videos, audio, multi-column, etc.)


# TODO(IMAGE_TEXT_SUPPORT): Add preprocessing config dataclasses here
# See /IMAGE_TEXT_SUPPORT.md for implementation guide
# Need: ImageAugmentationConfig, ImagePreprocessingConfig, TextPreprocessingConfig


# ============================================
# Core Data Models
# ============================================


@dataclass_json
@dataclass
class Metric:
    """
    Evaluation metric definition.

    Defines what to measure (not the measurement result).
    """

    name: str  # e.g., "f1_score", "accuracy", "rmse", "mae", "roc_auc"
    optimization_direction: str  # "higher" (maximize) or "lower" (minimize)


@dataclass
class BuildContext:
    """
    Context passed through workflow phases.

    Holds all state for a model building run.
    """

    # Identifiers
    user_id: str
    experiment_id: str
    dataset_uri: str  # Training dataset URI (always Parquet after normalization)
    work_dir: Path  # Working directory for all artifacts
    intent: str  # User's natural language task description

    # Input format tracking (for multi-format support)
    input_format: str | None = None  # Original format (csv, orc, avro, parquet)
    input_format_options: dict[str, Any] = field(default_factory=dict)  # Format-specific options (e.g., CSV delimiter)

    # Data understanding phase
    data_layout: Optional["DataLayout"] = None  # Physical structure of dataset (flat_numeric, image_path, text_string)
    viable_model_types: list[str] = field(
        default_factory=list
    )  # Model types viable for this task (e.g., ["xgboost", "catboost"])
    primary_input_column: str | None = None  # For non-tabular: column with image paths or text content
    stats: dict[str, Any] | None = None
    task_analysis: dict[str, Any] | None = None
    metric: Optional["Metric"] = None
    compute_metric: Any | None = None  # Function for computing metric (callable)
    output_targets: list[str] = field(default_factory=list)  # Target column(s) identified by MLTaskAnalyser
    group_column: str | None = None  # For ranking: query_id, session_id, user_id (group identifier)

    # Data preparation phase
    train_uri: str | None = None
    val_uri: str | None = None
    test_uri: str | None = None
    train_sample_uri: str | None = None  # Large sample for pipeline fitting
    val_sample_uri: str | None = None  # Small sample for quick evaluation

    # Feature engineering phase
    feature_pipeline: Pipeline | None = None
    train_transformed_uri: str | None = None
    val_transformed_uri: str | None = None
    test_transformed_uri: str | None = None

    # Baseline phase
    baseline_predictor: Any | None = None  # Object with .predict() method
    baseline_performance: float | None = None
    heuristic_baseline: Optional["Baseline"] = None  # Legacy - for comparison in outer loop

    # Search infrastructure
    insight_store: Any | None = None  # InsightStore for accumulating learnings during search

    # Outer loop feedback
    feedback: list[dict[str, Any]] = field(default_factory=list)

    # Scratch space for passing objects between agents
    scratch: dict[str, Any] = field(default_factory=dict)

    def add_outer_loop_feedback(self, solution: Optional["Solution"], issue: str):
        """Add feedback for outer loop retry."""
        self.feedback.append({"iteration": len(self.feedback), "issue": issue, "solution": solution})

    def update(self, **kwargs):
        """Convenience method to update multiple fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"BuildContext has no attribute '{key}'")

    def to_dict(self) -> dict:
        """
        Serialize BuildContext to dict for checkpointing.

        Handles non-serializable objects by skipping or converting them.
        """
        return {
            # Identifiers
            "user_id": self.user_id,
            "experiment_id": self.experiment_id,
            "dataset_uri": self.dataset_uri,
            "work_dir": str(self.work_dir),
            "intent": self.intent,
            # Input format tracking
            "input_format": self.input_format,
            "input_format_options": self.input_format_options,
            # Phase 1 fields
            "data_layout": self.data_layout.value if self.data_layout else None,
            "viable_model_types": self.viable_model_types,
            "primary_input_column": self.primary_input_column,
            "stats": self.stats,
            "task_analysis": self.task_analysis,
            "metric": self.metric.to_dict() if self.metric else None,
            "compute_metric": (
                inspect.getsource(self.compute_metric)
                if self.compute_metric and callable(self.compute_metric)
                else None
            ),
            "output_targets": self.output_targets,
            "group_column": self.group_column,
            # Phase 2 fields
            "train_uri": self.train_uri,
            "val_uri": self.val_uri,
            "test_uri": self.test_uri,
            "train_sample_uri": self.train_sample_uri,
            "val_sample_uri": self.val_sample_uri,
            # Feature engineering fields
            "train_transformed_uri": self.train_transformed_uri,
            "val_transformed_uri": self.val_transformed_uri,
            "test_transformed_uri": self.test_transformed_uri,
            # Phase 3 fields
            "heuristic_baseline": self.heuristic_baseline.to_dict() if self.heuristic_baseline else None,
            "baseline_performance": self.baseline_performance,
            # Skipped: feature_pipeline (saved separately), baseline_predictor (runtime only)
            # Skipped: insight_store (serialized separately)
            # Feedback and scratch
            "feedback": self.feedback,
            "scratch": {k: v for k, v in self.scratch.items() if isinstance(v, str | int | float | bool | dict | list)},
        }

    @staticmethod
    def from_dict(d: dict) -> "BuildContext":
        """Deserialize BuildContext from checkpoint dict."""
        # Handle compute_metric (source code string)
        compute_metric = None
        if d.get("compute_metric"):
            try:
                exec_globals = {}
                exec(d["compute_metric"], exec_globals)
                # Find the function in exec_globals
                compute_metric = next((v for v in exec_globals.values() if callable(v)), None)
            except Exception:
                compute_metric = None

        return BuildContext(
            user_id=d["user_id"],
            experiment_id=d["experiment_id"],
            dataset_uri=d["dataset_uri"],
            work_dir=Path(d["work_dir"]),
            intent=d["intent"],
            input_format=d.get("input_format"),
            input_format_options=d.get("input_format_options", {}),
            data_layout=DataLayout(d["data_layout"]) if d.get("data_layout") else None,
            viable_model_types=d.get("viable_model_types", []),
            primary_input_column=d.get("primary_input_column"),
            stats=d.get("stats"),
            task_analysis=d.get("task_analysis"),
            metric=Metric.from_dict(d["metric"]) if d.get("metric") else None,
            compute_metric=compute_metric,
            output_targets=d.get("output_targets", []),
            group_column=d.get("group_column"),
            train_uri=d.get("train_uri"),
            val_uri=d.get("val_uri"),
            test_uri=d.get("test_uri"),
            train_sample_uri=d.get("train_sample_uri"),
            val_sample_uri=d.get("val_sample_uri"),
            train_transformed_uri=d.get("train_transformed_uri"),
            val_transformed_uri=d.get("val_transformed_uri"),
            test_transformed_uri=d.get("test_transformed_uri"),
            heuristic_baseline=Baseline.from_dict(d["heuristic_baseline"]) if d.get("heuristic_baseline") else None,
            baseline_performance=d.get("baseline_performance"),
            feedback=d.get("feedback", []),
            scratch=d.get("scratch", {}),
        )


@dataclass_json
@dataclass
class Baseline:
    """
    Represents a baseline model result.
    """

    name: str  # "heuristic_most_frequent", "autogluon", etc.
    model_type: str  # Type of baseline
    performance: float  # Metric value
    model_artifacts_path: Path | None = None  # Path to saved model
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Baseline(name={self.name}, performance={self.performance:.4f})"


@dataclass
class Solution:
    """
    Represents a solution in the search tree.

    Each solution contains:
    - Feature engineering (sklearn Pipeline)
    - Model (untrained initially, then trained)
    - Training results (artifacts, performance)
    - Tree structure (parent/children for search)

    TODO: Add transformed_schema field to store AUTHORITATIVE schema
        This should contain:
        - columns: List[str] with meaningful names (e.g., "HomePlanet_Earth", "Age_scaled")
        - dtypes: Dict[str, str]
        - num_features: int
        - feature_mapping: Dict tracing output → original columns
        Currently schema is extracted ad-hoc in workflow; should be part of Solution.
    """

    # Required Fields
    solution_id: int  # Unique identifier for this solution node
    feature_pipeline: Pipeline
    model: Any  # XGBClassifier, keras.Sequential, etc. (untrained initially)
    model_type: str  # "xgboost", "keras" - determines training template

    # Neural network training params (populated for keras and pytorch model types)
    optimizer: Any | None = None  # Optimizer instance (Keras or PyTorch)
    loss: Any | None = None  # Loss instance (Keras or PyTorch)
    epochs: int | None = None  # Training epochs
    batch_size: int | None = None  # Batch size

    # Execution Results
    model_artifacts_path: Path | None = None
    performance: float | None = None
    training_time: float | None = None

    # Tree Structure
    parent: Optional["Solution"] = None
    children: list["Solution"] = field(default_factory=list)

    # Search Metadata
    stage: str = "draft"  # 'draft', 'debug', or 'improve'
    plan: Optional["UnifiedPlan"] = None  # Structured plan specification

    # Error Tracking
    error: str | None = None
    is_buggy: bool = False

    # ============================================
    # Properties
    # ============================================

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node in the search tree."""
        return len(self.children) == 0

    @property
    def debug_depth(self) -> int:
        """
        Number of consecutive buggy ancestors in lineage.

        Used to prevent infinite debug loops - tracks how many buggy
        solutions are in the parent chain.
        """
        if not self.is_buggy:
            return 0
        return (self.parent.debug_depth + 1) if self.parent else 1

    @property
    def is_successful(self) -> bool:
        """Check if execution succeeded."""
        return not self.is_buggy and self.performance is not None

    def __repr__(self) -> str:
        stage = "root" if self.parent is None else "child"
        if self.is_buggy:
            return f"Solution(id={self.solution_id}, {stage}, BUGGY)"
        elif self.performance is not None:
            return f"Solution(id={self.solution_id}, {stage}, perf={self.performance:.4f})"
        else:
            return f"Solution(id={self.solution_id}, {stage}, pending)"

    def to_dict(self) -> dict:
        """
        Serialize solution to dict for checkpointing.

        Handles circular parent/child references by storing solution IDs.
        Skips unpicklable objects (pipeline, model) - these are saved separately to disk.
        """
        from plexe.checkpointing import pickle_to_base64

        return {
            "solution_id": self.solution_id,
            "model_type": self.model_type,
            "parent_solution_id": self.parent.solution_id if self.parent else None,
            "child_solution_ids": [c.solution_id for c in self.children],
            "performance": self.performance,
            "training_time": self.training_time,
            "is_buggy": self.is_buggy,
            "error": self.error,
            "stage": self.stage,
            "model_artifacts_path": str(self.model_artifacts_path) if self.model_artifacts_path else None,
            # Keras-specific fields (pickle to base64)
            "optimizer": pickle_to_base64(self.optimizer) if self.optimizer else None,
            "loss": pickle_to_base64(self.loss) if self.loss else None,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            # Plan (already has to_dict via @dataclass_json)
            "plan": self.plan.to_dict() if self.plan else None,
        }

    @staticmethod
    def from_dict(d: dict, all_solutions: dict[int, "Solution"]) -> "Solution":
        """
        Deserialize solution from checkpoint dict.

        Args:
            d: Solution dict from checkpoint
            all_solutions: Dict mapping solution ID → Solution (for rebuilding parent/child links)

        Returns:
            Solution instance (parent/children will be None initially, linked in second pass)
        """
        from plexe.checkpointing import base64_to_pickle

        # Reconstruct plan if present
        plan = UnifiedPlan.from_dict(d["plan"]) if d.get("plan") else None

        # Create solution without parent/children (will be linked later)
        return Solution(
            solution_id=d["solution_id"],
            feature_pipeline=None,  # type: ignore - Will be loaded from disk if needed
            model=None,  # type: ignore - Will be loaded from disk if needed
            model_type=d["model_type"],
            model_artifacts_path=Path(d["model_artifacts_path"]) if d.get("model_artifacts_path") else None,
            performance=d.get("performance"),
            training_time=d.get("training_time"),
            parent=None,  # Will be linked in second pass
            children=[],  # Will be linked in second pass
            stage=d.get("stage", "draft"),
            plan=plan,
            error=d.get("error"),
            is_buggy=d.get("is_buggy", False),
            # Keras-specific
            optimizer=base64_to_pickle(d["optimizer"]) if d.get("optimizer") else None,
            loss=base64_to_pickle(d["loss"]) if d.get("loss") else None,
            epochs=d.get("epochs"),
            batch_size=d.get("batch_size"),
        )


# ============================================
# Hypothesis-Driven Search Models
# ============================================


@dataclass_json
@dataclass
class Insight:
    """
    Structured learning extracted from search experiments.

    Captures what we learned from trying variations.
    """

    id: int  # Unique insight ID
    change: str  # What was varied (e.g., "n_estimators: 100→500")
    effect: str  # Observed outcome (e.g., "+5.8% improvement, peak at 250")
    context: str  # When this applies (e.g., "for datasets with ~8k rows, ~13 features")
    confidence: Literal["high", "medium", "low"]  # How confident are we?
    supporting_evidence: list[int]  # Solution IDs that support this
    timestamp: str = ""  # When extracted


@dataclass_json
@dataclass
class Hypothesis:
    """
    Strategic direction for next exploration.

    Generated by HypothesiserAgent based on insights and history.
    """

    expand_solution_id: int  # Which solution to build upon
    focus: Literal["features", "model", "both"]  # What to vary
    vary: str  # What specifically to change (e.g., "n_estimators", "scaling_strategy")
    num_variants: int  # How many variations to try
    rationale: str  # Why this direction is promising
    keep_from_parent: list[str]  # What to reuse (e.g., ["features"] or ["model"])
    expected_impact: str  # Predicted effect (e.g., "±3-5% performance swing expected")


@dataclass_json
@dataclass
class FeaturePlan:
    """
    Feature engineering specification.
    """

    strategy: Literal["reuse_parent", "new", "modify_parent"]
    parent_solution_id: int | None = None  # If reusing/modifying parent
    changes: dict[str, Any] = field(default_factory=dict)  # Specific changes to make
    rationale: str = ""  # Why this feature strategy


@dataclass_json
@dataclass
class ModelPlan:
    """
    Model configuration specification (natural language directive).
    """

    model_type: str  # "xgboost", "keras" - per-plan model architecture choice
    directive: str = ""  # What to do (e.g., "Increase tree count to around 250, keep other params similar")
    change_summary: str = ""  # What changed from parent (e.g., "n_estimators: 100→~250")
    rationale: str = ""  # Why this change


@dataclass_json
@dataclass
class UnifiedPlan:
    """
    Complete solution specification (features + model).

    Prescriptive plan that guides implementation.
    """

    variant_id: str  # "A", "B", "C" for multi-variant hypotheses
    parent_solution_id: int  # Which solution this builds upon
    features: FeaturePlan
    model: ModelPlan
    hypothesis_rationale: str  # Why we're trying this (from Hypothesis)
    expected_outcome: str  # What we expect to learn


# ============================================
# Evaluation Reports
# ============================================


@dataclass_json
@dataclass
class CoreMetricsReport:
    """
    Core performance metrics on test set.

    Phase 1 of evaluation: comprehensive metric computation.
    """

    task_type: str  # "binary_classification", "multiclass_classification", "regression"
    primary_metric_name: str  # Name of primary optimization metric
    primary_metric_value: float  # Value of primary metric
    primary_metric_ci_lower: float | None  # 95% CI lower bound (optional but encouraged)
    primary_metric_ci_upper: float | None  # 95% CI upper bound (optional but encouraged)
    all_metrics: dict[str, float]  # {metric_name: value}
    metric_confidence_intervals: dict[str, tuple[float, float]] | None  # Optional CIs for other metrics
    statistical_notes: str  # Agent's interpretation
    visualizations: dict[str, str] | None  # {plot_name: base64_png} - optional


@dataclass_json
@dataclass
class DiagnosticReport:
    """
    Error analysis and failure pattern detection.

    Phase 2 of evaluation: understanding where and why model fails.
    """

    worst_predictions: list[dict]  # [{index, true_value, predicted_value, error, features}]
    error_patterns: list[str]  # Human-readable patterns
    subgroup_analysis: dict[str, dict] | None  # {subgroup_name: {metrics}} - optional
    key_insights: list[str]  # Actionable insights
    error_distribution_summary: str  # Agent's summary


@dataclass_json
@dataclass
class RobustnessReport:
    """
    Model reliability under stress conditions.

    Phase 3 of evaluation: perturbation testing and consistency checks.
    """

    perturbation_tests: dict[str, dict]  # {test_name: {results}}
    consistency_score: float | None  # 0-1 if computed
    robustness_grade: str  # "A", "B", "C", "D", "F"
    concerns: list[str]  # Identified risks
    recommendations: list[str]  # Mitigation suggestions


@dataclass_json
@dataclass
class ExplainabilityReport:
    """
    Feature importance and model interpretability.

    Phase 4 of evaluation (conditional): understanding what drives predictions.
    """

    feature_importance: dict[str, float]  # Feature -> importance
    method_used: str  # Method for computing importance
    top_features: list[str]  # Most important features
    confidence_intervals: dict[str, tuple[float, float]] | None  # Optional CIs for feature importance
    interpretation: str  # What do these features mean?


@dataclass_json
@dataclass
class BaselineComparisonReport:
    """
    Model vs. baseline performance comparison.

    Phase 4.5 of evaluation: contextualizing performance gains.
    """

    baseline_name: str
    baseline_type: str  # "heuristic", "statistical", "simple_model"
    baseline_description: str
    baseline_performance: dict[str, float]  # {metric_name: value}
    model_performance: dict[str, float]  # {metric_name: value}
    performance_delta: dict[str, float]  # {metric_name: absolute_difference}
    performance_delta_pct: dict[str, float] | None  # {metric_name: percentage_change}
    interpretation: str  # Agent's interpretation of comparison


@dataclass_json
@dataclass
class EvaluationReport:
    """
    Final comprehensive evaluation with verdict and recommendations.

    Phase 5 of evaluation (synthesis): decision-making for deployment.
    """

    # Executive Summary
    verdict: Literal["PASS", "CONDITIONAL_PASS", "FAIL"]
    summary: str
    deployment_ready: bool
    key_concerns: list[str]

    # Component Reports
    core_metrics: CoreMetricsReport
    diagnostics: DiagnosticReport
    robustness: RobustnessReport
    explainability: ExplainabilityReport | None
    baseline_comparison: BaselineComparisonReport

    # Overall Recommendations
    recommendations: list[dict]  # [{priority: "HIGH|MEDIUM|LOW", action: str, rationale: str}]


# ============================================
# Training Errors
# ============================================


class TrainingError(Exception):
    """Raised when training fails."""

    pass


class ValidationError(Exception):
    """Raised when validation fails."""

    pass
