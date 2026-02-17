"""
Main workflow orchestrator.

Coordinates 6-phase sequential ML model building pipeline:
1. Data Understanding (stats, task analysis, metric selection)
2. Data Preparation (splitting, sampling)
3. Baseline Models (simple heuristic baseline)
4. Model Search (iterative tree-search)
5. Final Evaluation (test set metrics)
6. Packaging (consolidate all deliverables)
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import shutil
import sys
import tarfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import pandas as pd
import yaml

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.integrations.base import WorkflowIntegration
from plexe.config import Config
from plexe.constants import DirNames, PhaseNames
from plexe.models import BuildContext, Solution, Baseline, Hypothesis, DataLayout, EvaluationReport
from plexe.execution.training.runner import TrainingRunner
from plexe.checkpointing import save_checkpoint, load_checkpoint
from plexe.search.journal import SearchJournal
from plexe.search.policy import SearchPolicy
from plexe.search.insight_store import InsightStore

from plexe.agents.hypothesiser import HypothesiserAgent
from plexe.agents.planner import PlannerAgent
from plexe.agents.insight_extractor import InsightExtractorAgent
from plexe.agents.layout_detector import LayoutDetectionAgent
from plexe.agents.statistical_analyser import StatisticalAnalyserAgent
from plexe.agents.ml_task_analyser import MLTaskAnalyserAgent
from plexe.agents.metric_selector import MetricSelectorAgent
from plexe.agents.metric_implementer import MetricImplementationAgent
from plexe.agents.dataset_splitter import DatasetSplitterAgent
from plexe.agents.sampler import SamplingAgent
from plexe.agents.baseline_builder import BaselineBuilderAgent
from plexe.agents.feature_processor import FeatureProcessorAgent
from plexe.agents.model_definer import ModelDefinerAgent
from plexe.agents.model_evaluator import ModelEvaluatorAgent
from plexe.utils.tracing import tracer
from plexe.utils.reporting import save_report
from plexe.templates.features.pipeline_fitter import fit_pipeline
from plexe.templates.features.pipeline_runner import transform_dataset_via_spark
from plexe.helpers import evaluate_on_sample, select_viable_model_types

logger = logging.getLogger(__name__)


# TODO: multi-dataset joining logic is not supported yet, but is a missing feature compared to old
#  model-builder. Need to add this in future.


# ============================================
# Main Orchestrator
# ============================================


def build_model(
    spark: SparkSession,
    train_dataset_uri: str,
    test_dataset_uri: str | None,
    user_id: str,
    intent: str,
    experiment_id: str,
    work_dir: Path,
    runner: TrainingRunner,
    search_policy: SearchPolicy,
    config: Config,
    integration: WorkflowIntegration,
    enable_final_evaluation: bool = False,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
    pause_points: list[str] | None = None,
    on_pause: Callable[[str], None] | None = None,
    user_feedback: dict | None = None,
) -> tuple[Solution, dict, EvaluationReport | None] | None:
    """
    Main workflow orchestrator.

    Args:
        spark: SparkSession
        train_dataset_uri: URI to training dataset
        test_dataset_uri: Optional URI to separate test dataset
        user_id: User identifier
        intent: Natural language description of ML task
        experiment_id: Experiment identifier
        work_dir: Working directory for all artifacts
        runner: Training runner (local, SageMaker, etc.)
        search_policy: Search policy (agent-driven, etc.)
        config: Configuration
        integration: WorkflowIntegration for infrastructure queries
        enable_final_evaluation: Whether to run final test set evaluation
        on_checkpoint_saved: Optional callback(phase_name, checkpoint_path, work_dir) for external sync
        pause_points: Optional list of phase names to pause at for user feedback
                     (e.g., ["01_analyze_data", "04_search_models"])
        on_pause: Optional callback(phase_name) called when workflow pauses
        user_feedback: User feedback dict when resuming from pause (optional)

    Returns:
        (best_solution, final_metrics, evaluation_report) tuple if completed
        None if workflow paused for feedback
    """

    logger.info(f"Starting model building workflow for experiment {experiment_id}")
    logger.info(f"User intent: {intent}")

    # ============================================
    # Check for Existing Checkpoints (Resume Logic)
    # ============================================
    start_phase = 1
    context = None
    journal = None
    insight_store = None

    # Try to load most recent checkpoint (check phases in reverse order)
    phase_checkpoints = [
        (6, PhaseNames.PACKAGE_FINAL_MODEL),
        (5, PhaseNames.EVALUATE_FINAL),
        (4, PhaseNames.SEARCH_MODELS),
        (3, PhaseNames.BUILD_BASELINES),
        (2, PhaseNames.PREPARE_DATA),
        (1, PhaseNames.ANALYZE_DATA),
    ]

    for phase_num, phase_name in phase_checkpoints:
        checkpoint_data = load_checkpoint(phase_name, work_dir)
        if checkpoint_data:
            logger.info(f"Found checkpoint: {phase_name} - resuming from phase {phase_num + 1}")

            # Deserialize state
            context = BuildContext.from_dict(checkpoint_data["context"])
            if checkpoint_data.get("search_journal"):
                journal = SearchJournal.from_dict(checkpoint_data["search_journal"])
                logger.info(f"Restored SearchJournal with {len(journal.nodes)} solutions")
            if checkpoint_data.get("insight_store"):
                insight_store = InsightStore.from_dict(checkpoint_data["insight_store"])
                context.insight_store = insight_store
                logger.info(f"Restored InsightStore with {len(insight_store)} insights")

            # Check for user feedback in checkpoint (for resume after pause)
            if checkpoint_data.get("user_feedback"):
                context.scratch["_user_feedback"] = checkpoint_data["user_feedback"]
                logger.info("📝 Restored user feedback from paused checkpoint")

            start_phase = phase_num + 1
            break

    # If no checkpoint found, create fresh context
    if context is None:
        logger.info("No checkpoint found - starting from beginning")
        context = BuildContext(
            user_id=user_id,
            experiment_id=experiment_id,
            dataset_uri=train_dataset_uri,
            work_dir=work_dir,
            intent=intent,
        )

    # Inject user feedback if provided (for resume_with_feedback flow)
    # Agents will read this from context.scratch["_user_feedback"] and incorporate into their prompts
    if user_feedback:
        context.scratch["_user_feedback"] = user_feedback
        logger.info("📝 User feedback injected - agents will incorporate guidance into their work")

    # Wrap entire workflow in top-level trace span
    with tracer.start_as_current_span("ModelBuilder") as root_span:
        root_span.set_attribute("experiment_id", experiment_id)
        root_span.set_attribute("user_id", user_id)

        # Phase 1: Data Understanding
        if start_phase <= 1:
            with tracer.start_as_current_span("Phase 1: Data Understanding"):
                analyze_data(spark, train_dataset_uri, context, config, on_checkpoint_saved)

            # Check if should pause after this phase
            if pause_points and PhaseNames.ANALYZE_DATA in pause_points:
                logger.info("⏸️  Workflow paused at Phase 1 for user feedback")
                if on_pause:
                    on_pause(PhaseNames.ANALYZE_DATA)
                return None
        else:
            logger.info("Skipping Phase 1 (already completed)")

        # Phase 2: Data Preparation
        if start_phase <= 2:
            with tracer.start_as_current_span("Phase 2: Data Preparation"):
                # Use sanitized dataset if available (from Phase 1 column name cleaning)
                # If resuming from checkpoint and sanitized URI not in context, use original
                # (Phase 1 checkpoint should have stored the sanitized URI)
                train_uri_to_use = context.scratch.get("_sanitized_dataset_uri", train_dataset_uri)

                # Sanitize test dataset too if provided (must match training data schema)
                test_uri_to_use = test_dataset_uri
                if test_dataset_uri and "_original_column_names" in context.scratch:
                    import re

                    # Reuse training data's column mapping to ensure consistency
                    column_mapping = context.scratch["_original_column_names"]
                    test_df = spark.read.parquet(test_dataset_uri)

                    # Apply same transformations as training data
                    for original, sanitized in column_mapping.items():
                        if original in test_df.columns:
                            # Check if target name already exists (would be silently overwritten)
                            if sanitized in test_df.columns and sanitized != original:
                                raise ValueError(
                                    f"Cannot sanitize test dataset: column '{original}' would rename to "
                                    f"'{sanitized}', but '{sanitized}' already exists in test dataset. "
                                    f"This would cause data loss. Please ensure test dataset columns match "
                                    f"training dataset or have unique sanitized names."
                                )
                            test_df = test_df.withColumnRenamed(original, sanitized)

                    # Handle columns in test that weren't in training data
                    test_only_mapping = {}
                    # Track all existing names: training-sanitized + already-processed test columns
                    all_existing_names = set(column_mapping.values())

                    for idx, col_name in enumerate(test_df.columns):
                        if col_name not in column_mapping.values() and any(
                            char in col_name for char in [".", " ", "-", "(", ")", "[", "]"]
                        ):
                            safe_name = re.sub(r"[.\s\-\(\)\[\]]", "_", col_name)
                            safe_name = re.sub(r"_+", "_", safe_name).strip("_")
                            if not safe_name:
                                safe_name = f"col_{idx}"

                            # Check for collisions with training names AND other test columns
                            original_safe_name = safe_name
                            counter = 1
                            while safe_name in all_existing_names:
                                safe_name = f"{original_safe_name}_{counter}"
                                counter += 1

                            test_df = test_df.withColumnRenamed(col_name, safe_name)
                            test_only_mapping[col_name] = safe_name
                            all_existing_names.add(safe_name)  # Mark as used
                            logger.info(f"  Test-only column: '{col_name}' → '{safe_name}'")

                    # Save sanitized test dataset
                    test_uri_to_use = f"{context.work_dir}/{DirNames.BUILD_DIR}/data/test_sanitized.parquet"
                    test_df.write.mode("overwrite").parquet(test_uri_to_use)
                    logger.info("✓ Test dataset sanitized using training column mapping")
                    context.scratch["_sanitized_test_dataset_uri"] = test_uri_to_use

                prepare_data(
                    spark,
                    train_uri_to_use,
                    test_uri_to_use,
                    context,
                    config,
                    integration,
                    enable_final_evaluation,
                    on_checkpoint_saved,
                )

            # Check if should pause after this phase
            if pause_points and PhaseNames.PREPARE_DATA in pause_points:
                logger.info("⏸️  Workflow paused at Phase 2 for user feedback")
                if on_pause:
                    on_pause(PhaseNames.PREPARE_DATA)
                return None
        else:
            logger.info("Skipping Phase 2 (already completed)")

        # Phase 3: Baseline Models
        if start_phase <= 3:
            with tracer.start_as_current_span("Phase 3: Baseline Models"):
                build_baselines(spark, context, config, on_checkpoint_saved)

            # Check if should pause after this phase
            if pause_points and PhaseNames.BUILD_BASELINES in pause_points:
                logger.info("⏸️  Workflow paused at Phase 3 for user feedback")
                if on_pause:
                    on_pause(PhaseNames.BUILD_BASELINES)
                return None
        else:
            logger.info("Skipping Phase 3 (already completed)")

        # Phase 4: Model Search (Iterative)
        if start_phase <= 4:
            with tracer.start_as_current_span("Phase 4: Model Search"):
                best_solution = search_models(
                    spark,
                    context,
                    runner,
                    search_policy,
                    config,
                    integration,
                    on_checkpoint_saved,
                    journal,
                    insight_store,
                )

            # Check if should pause after this phase
            if pause_points and PhaseNames.SEARCH_MODELS in pause_points:
                logger.info("⏸️  Workflow paused at Phase 4 for user feedback")
                if on_pause:
                    on_pause(PhaseNames.SEARCH_MODELS)
                return None
        else:
            logger.info("Skipping Phase 4 (already completed)")
            # When resuming after Phase 4, we need to extract best_solution from restored journal
            if journal and journal.best_node:
                best_solution = journal.best_node
                logger.info(f"Restored best solution from checkpoint: solution_id={best_solution.solution_id}")
            else:
                raise RuntimeError("Cannot resume after Phase 4 without valid SearchJournal")

        # Handle case where no successful solution found
        if not best_solution:
            logger.warning("No successful solutions found during search. Falling back to heuristic baseline.")
            best_solution = _baseline_to_solution(context.heuristic_baseline, context.work_dir)
            # Skip evaluation - baseline performance already known from Phase 3
            final_metrics = {
                "metric": context.metric.name,
                "performance": context.heuristic_baseline.performance,
                "test_samples": 0,
                "note": "Baseline fallback - no search solutions succeeded",
            }
        else:
            # Phase 5: Final Evaluation on Test Set (optional)
            if enable_final_evaluation and start_phase <= 5:
                with tracer.start_as_current_span("Phase 5: Final Evaluation"):
                    eval_metrics = evaluate_final(spark, context, best_solution, config, on_checkpoint_saved)
                    if eval_metrics:
                        final_metrics = eval_metrics
                    else:
                        # Fallback if evaluation fails
                        logger.warning("Evaluation failed - falling back to validation performance")
                        final_metrics = {
                            "metric": context.metric.name,
                            "performance": best_solution.performance,
                            "test_samples": 0,
                            "note": "Validation performance (evaluation failed)",
                        }

                # Check if should pause after this phase
                if pause_points and PhaseNames.EVALUATE_FINAL in pause_points:
                    logger.info("⏸️  Workflow paused at Phase 5 for user feedback")
                    if on_pause:
                        on_pause(PhaseNames.EVALUATE_FINAL)
                    return None
            elif start_phase > 5 and enable_final_evaluation:
                logger.info("Skipping Phase 5 (already completed)")
                # Use metrics from checkpoint or default
                final_metrics = {
                    "metric": context.metric.name,
                    "performance": best_solution.performance,
                    "note": "Loaded from checkpoint",
                }
            else:
                # Use validation performance from search
                final_metrics = {
                    "metric": context.metric.name,
                    "performance": best_solution.performance,
                    "validation_samples": "from search",
                    "note": "Validation performance (test evaluation disabled)",
                }

        # Fallback Logic: If primary model failed evaluation, try second-best
        evaluation_report = context.scratch.get("_evaluation_report")
        if evaluation_report and hasattr(evaluation_report, "verdict") and evaluation_report.verdict == "FAIL":
            logger.warning("=" * 80)
            logger.warning(f"PRIMARY MODEL FAILED EVALUATION: {evaluation_report.summary}")
            logger.warning("Attempting fallback to next-best valid solution...")
            logger.warning("=" * 80)

            # Save original evaluation report in case fallback also fails
            original_evaluation_report = evaluation_report
            original_final_metrics = final_metrics

            # Get journal from context (stored by search_models)
            journal = context.scratch.get("_search_journal")

            # Find all valid alternative solutions from journal
            if journal and journal.good_nodes:
                # Filter out the failed solution and any buggy ones
                valid_alternatives = [
                    sol
                    for sol in journal.good_nodes
                    if sol.solution_id != best_solution.solution_id and not sol.is_buggy
                ]

                if valid_alternatives:
                    # Sort by performance and pick the best alternative
                    valid_alternatives.sort(key=lambda s: s.performance, reverse=True)
                    fallback_solution = valid_alternatives[0]

                    logger.info(f"Found {len(valid_alternatives)} valid alternatives")
                    logger.info(
                        f"Best alternative: Solution {fallback_solution.solution_id} with {fallback_solution.performance:.4f} performance"
                    )
                    logger.info("Re-evaluating fallback solution...")

                    # Re-evaluate the fallback solution
                    with tracer.start_as_current_span("Fallback Solution Re-evaluation"):
                        fallback_eval_metrics = evaluate_final(
                            spark, context, fallback_solution, config, on_checkpoint_saved
                        )

                        if fallback_eval_metrics:
                            # Check if fallback passed evaluation
                            fallback_evaluation_report = context.scratch.get("_evaluation_report")

                            if fallback_evaluation_report and fallback_evaluation_report.verdict in [
                                "PASS",
                                "CONDITIONAL_PASS",
                            ]:
                                logger.info("=" * 80)
                                logger.info("✅ FALLBACK MODEL PASSED EVALUATION!")
                                logger.info(f"Verdict: {fallback_evaluation_report.verdict}")
                                logger.info(f"Performance: {fallback_solution.performance:.4f}")
                                logger.info(
                                    f"Switching from solution {best_solution.solution_id} (FAILED) to solution {fallback_solution.solution_id} (PASSED)"
                                )
                                logger.info("=" * 80)

                                # Use the fallback solution
                                best_solution = fallback_solution
                                final_metrics = fallback_eval_metrics
                                final_metrics["fallback_used"] = True
                                final_metrics["fallback_reason"] = "Primary model failed validation"
                                final_metrics["original_solution_id"] = (
                                    journal.best_node.solution_id if journal.best_node else None
                                )
                                final_metrics["fallback_solution_id"] = fallback_solution.solution_id
                            else:
                                logger.warning(
                                    f"Fallback solution also failed evaluation with verdict: {fallback_evaluation_report.verdict if fallback_evaluation_report else 'N/A'}"
                                )
                                logger.warning("Proceeding with original solution (will be marked as FAILED)")
                                # Restore original evaluation report (fallback overwrote it)
                                context.scratch["_evaluation_report"] = original_evaluation_report
                                final_metrics = original_final_metrics
                        else:
                            logger.warning("Fallback solution evaluation failed")
                            logger.warning("Proceeding with original solution (will be marked as FAILED)")
                            # Restore original evaluation report
                            context.scratch["_evaluation_report"] = original_evaluation_report
                            final_metrics = original_final_metrics
                else:
                    logger.error("No valid alternative solutions available for fallback")
                    logger.error("Proceeding with failed model (will be marked as FAILED)")
            else:
                logger.error("No search journal or good nodes available for fallback")
                logger.error("Proceeding with failed model (will be marked as FAILED)")

        # Phase 6: Package Final Deliverables
        if start_phase <= 6:
            with tracer.start_as_current_span("Phase 6: Package Final Model"):
                final_package_dir = package_final_model(
                    spark, context, best_solution, final_metrics, on_checkpoint_saved
                )

            # Check if should pause after this phase
            if pause_points and PhaseNames.PACKAGE_FINAL_MODEL in pause_points:
                logger.info("⏸️  Workflow paused at Phase 6 for user feedback")
                if on_pause:
                    on_pause(PhaseNames.PACKAGE_FINAL_MODEL)
                return None
        else:
            logger.info("Skipping Phase 6 (already completed)")
            final_package_dir = work_dir / "model"

        logger.info(f"Model building complete! Validation performance: {final_metrics}")
        logger.info(f"Final model package: {final_package_dir}")

        # Extract evaluation report from context (may be None if evaluation not run)
        evaluation_report = context.scratch.get("_evaluation_report")

        return best_solution, final_metrics, evaluation_report


# ============================================
# Phase 1: Data Understanding
# ============================================


def sanitize_dataset_column_names(spark: SparkSession, dataset_uri: str, context: BuildContext) -> str:
    """
    Sanitize column names by replacing special characters with underscores.

    Args:
        spark: SparkSession
        dataset_uri: Original dataset URI
        context: Build context

    Returns:
        URI of sanitized dataset (or original if no changes needed)
    """
    import re

    # Load dataset
    df = spark.read.parquet(dataset_uri)

    # Check if any columns need sanitization
    needs_sanitization = False
    column_mapping = {}

    for idx, col_name in enumerate(df.columns):
        # Replace special chars: dots, spaces, hyphens, parentheses, brackets
        safe_name = re.sub(r"[.\s\-\(\)\[\]]", "_", col_name)
        # Remove consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name)
        # Remove leading/trailing underscores
        safe_name = safe_name.strip("_")

        # Handle empty result (column was all special chars like "..." or "---")
        if not safe_name:
            safe_name = f"col_{idx}"

        # Handle collisions: if safe_name already used OR will collide with original, append counter
        if safe_name != col_name:
            original_safe_name = safe_name
            counter = 1
            # Check against ALL columns (originals + already assigned sanitized names)
            all_existing = set(df.columns) | set(column_mapping.values())
            while safe_name in all_existing:
                safe_name = f"{original_safe_name}_{counter}"
                counter += 1

            needs_sanitization = True
            column_mapping[col_name] = safe_name

    if not needs_sanitization:
        logger.info("Column names are clean - no sanitization needed")
        return dataset_uri

    # Rename columns
    logger.info(f"Sanitizing {len(column_mapping)} column names with special characters")
    for original, sanitized in column_mapping.items():
        logger.info(f"  '{original}' → '{sanitized}'")
        df = df.withColumnRenamed(original, sanitized)

    # Save sanitized dataset
    sanitized_uri = f"{context.work_dir}/{DirNames.BUILD_DIR}/data/dataset_sanitized.parquet"
    df.write.mode("overwrite").parquet(sanitized_uri)

    logger.info(f"✓ Sanitized dataset saved: {sanitized_uri}")

    # Store mapping for reference
    context.scratch["_original_column_names"] = column_mapping

    return sanitized_uri


def analyze_data(
    spark: SparkSession,
    dataset_uri: str,
    context: BuildContext,
    config: Config,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
):
    """
    Phase 1: Layout detection + Statistical + ML task analysis + metric selection.

    Args:
        spark: SparkSession
        dataset_uri: Dataset URI
        context: Build context
        config: Configuration
        on_checkpoint_saved: Optional callback for platform integration
    """

    logger.info("=== Phase 1: Data Understanding ===")

    # Step 0: Sanitize column names (handle special characters like dots)
    sanitized_uri = sanitize_dataset_column_names(spark, dataset_uri, context)
    context.scratch["_sanitized_dataset_uri"] = sanitized_uri

    # Step 1: Layout Detection (use sanitized dataset)
    layout_agent = LayoutDetectionAgent(spark, sanitized_uri, context, config)
    layout_info = layout_agent.run()
    save_report(context.work_dir, "00_layout_detection", layout_info)
    logger.info(
        f"Data layout detected: {context.data_layout.value}, primary_input_column={context.primary_input_column}"
    )

    # Validate layout is supported
    if context.data_layout == DataLayout.UNSUPPORTED:
        reason = layout_info.get("reason", "Unknown reason")
        error_msg = (
            f"Dataset layout is not supported.\n\n"
            f"Reason: {reason}\n\n"
            f"Supported layouts:\n"
            f"  - FLAT_NUMERIC: Traditional tabular data with numeric/categorical columns\n"
            f"  - IMAGE_PATH: Single column containing image file paths\n"
            f"  - TEXT_STRING: Single column containing text documents\n\n"
            f"Please restructure your data to match one of the supported layouts."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Step 0.5: Compute Viable Model Types
    # Three-tier filtering: selected frameworks → task-compatible → viable
    viable_model_types = select_viable_model_types(
        data_layout=context.data_layout, selected_frameworks=config.allowed_model_types
    )
    context.viable_model_types = viable_model_types
    logger.info(f"Viable model types for this task: {viable_model_types}")

    # Step 2: Statistical Analysis (use sanitized dataset)
    stats_agent = StatisticalAnalyserAgent(spark, sanitized_uri, context, config)
    stats = stats_agent.run()
    context.update(stats=stats)
    save_report(context.work_dir, "01_statistical_analysis", stats)

    # Step 3: ML Task Analysis (use sanitized dataset)
    task_agent = MLTaskAnalyserAgent(spark, sanitized_uri, stats, context, config)
    task_analysis = task_agent.run()
    context.update(task_analysis=task_analysis)
    save_report(context.work_dir, "02_task_analysis", task_analysis)
    # Note: output_targets already set by task_agent.run()

    # Step 3: Metric Selection
    metric_agent = MetricSelectorAgent(context, config)
    metric = metric_agent.run()
    context.update(metric=metric)
    # Save full selection with rationale (agent stores this in scratch)
    save_report(context.work_dir, "03_metric_selection", context.scratch.get("_metric_selection", {}))

    # Step 4: Metric Implementation (only if custom metric)
    from plexe.config import StandardMetric

    # Check if metric is standard (hardcoded) or custom (needs generation)
    standard_metric_values = [m.value for m in StandardMetric]
    if metric.name in standard_metric_values:
        logger.info(f"Using hardcoded implementation for standard metric: {metric.name}")
        context.compute_metric = None  # Signal: use hardcoded
    else:
        logger.info(f"Custom metric '{metric.name}' detected - generating implementation")
        metric_impl_agent = MetricImplementationAgent(context, config)
        metric_impl_agent.run()  # Sets context.compute_metric directly

    logger.info(
        f"Data understanding complete: {task_analysis.get('task_type')} task, targets={context.output_targets}, metric={metric.name}"
    )

    # Save checkpoint
    _save_phase_checkpoint(PhaseNames.ANALYZE_DATA, context, on_checkpoint_saved)


# ============================================
# Phase 2: Data Preparation
# ============================================


def prepare_data(
    spark: SparkSession,
    training_dataset_uri: str,
    test_dataset_uri: str | None,
    context: BuildContext,
    config: Config,
    integration: WorkflowIntegration,
    generate_test_set: bool,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
):
    """
    Phase 2: Split dataset and extract sample.

    Supports two modes:
    1. Single dataset: Split training_dataset_uri into train/val(/test if enabled)
    2. Separate datasets: Split training_dataset_uri into train/val, use test_dataset_uri as test

    Args:
        spark: SparkSession
        training_dataset_uri: URI to training dataset
        test_dataset_uri: Optional URI to separate test dataset
        context: Build context
        config: Configuration
        integration: WorkflowIntegration for infrastructure queries
        generate_test_set: Whether to create test set from training data (ignored if test_dataset_uri provided)
        on_checkpoint_saved: Optional callback for platform integration
    """

    logger.info("=== Phase 2: Data Preparation ===")

    # Step 1: Handle Test Dataset (if provided separately)
    if test_dataset_uri:
        logger.info(f"Separate test dataset mode: {test_dataset_uri}")

        # Copy test dataset to DirNames.BUILD_DIR/data/ for consistency
        test_df = spark.read.parquet(test_dataset_uri)
        test_uri = str(context.work_dir / DirNames.BUILD_DIR / "data" / "test.parquet")
        test_df.write.mode("overwrite").parquet(test_uri)
        logger.info(f"Copied test dataset to: {test_uri}")

        # Always 2-way split when separate test provided
        split_ratios = {"train": 0.85, "val": 0.15}
        logger.info("Splitting training data into train/val only (test provided separately)")

    else:
        # Single dataset mode: create test from split if requested
        test_uri = None
        if generate_test_set:
            # 3-way split: train/val/test
            # Handle both new schema (ratios nested) and legacy (ratios at top level)
            recommended_split = context.task_analysis.get("recommended_split", {})
            if "ratios" in recommended_split:
                # New schema: extract ratios from nested structure
                split_ratios = recommended_split["ratios"]
            elif "train" in recommended_split:
                # Legacy schema: ratios at top level
                split_ratios = recommended_split
            else:
                # Default fallback
                split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
            logger.info("Creating train/val/test splits from single dataset (final evaluation enabled)")
        else:
            # 2-way split: train/val only
            split_ratios = {"train": 0.85, "val": 0.15}
            logger.info("Creating train/val splits only (final evaluation disabled)")

    # Step 2: Split Training Dataset
    splitter = DatasetSplitterAgent(spark=spark, dataset_uri=training_dataset_uri, context=context, config=config)

    # Get splits output location from integration (based on dataset location)
    splits_output_dir = integration.get_artifact_location(
        "splits", training_dataset_uri, context.experiment_id, context.work_dir
    )

    train_uri, val_uri, split_test_uri = splitter.run(split_ratios=split_ratios, output_dir=splits_output_dir)

    # Use separate test if provided, otherwise use split test
    test_uri = test_uri if test_dataset_uri else split_test_uri

    # Step 3: Create Intelligent Samples
    sampler = SamplingAgent(spark=spark, context=context, config=config)

    samples_output_dir = integration.get_artifact_location(
        "samples", training_dataset_uri, context.experiment_id, context.work_dir
    )

    train_sample_uri, val_sample_uri = sampler.run(
        train_uri=train_uri,
        val_uri=val_uri,
        train_sample_size=config.train_sample_size,
        val_sample_size=config.val_sample_size,
        output_dir=samples_output_dir,
    )

    # Step 3b: Ensure samples are local (download from S3 if needed)
    train_sample_uri, val_sample_uri = integration.ensure_local([train_sample_uri, val_sample_uri], context.work_dir)

    # Step 4: Update Context
    context.update(
        train_uri=train_uri,
        val_uri=val_uri,
        test_uri=test_uri,
        train_sample_uri=train_sample_uri,
        val_sample_uri=val_sample_uri,
    )

    # Step 5: Validate schema compatibility (if separate test provided)
    if test_dataset_uri:
        logger.info("Validating test dataset schema compatibility...")
        train_df = spark.read.parquet(train_uri)
        test_df = spark.read.parquet(test_uri)

        # Check target column exists in test
        if context.output_targets[0] not in test_df.columns:
            raise ValueError(
                f"Test dataset missing target column '{context.output_targets[0]}'. " f"Test columns: {test_df.columns}"
            )

        # Check feature overlap
        train_features = set(train_df.columns) - set(context.output_targets)
        test_features = set(test_df.columns) - set(context.output_targets)

        missing_in_test = train_features - test_features
        if missing_in_test:
            raise ValueError(
                f"Test dataset missing {len(missing_in_test)} features from training data: {sorted(missing_in_test)}\n"
                f"Model cannot make predictions without these features."
            )

        extra_in_test = test_features - train_features
        if extra_in_test:
            logger.warning(
                f"Test dataset has {len(extra_in_test)} extra features not in training data (will be ignored)"
            )

        logger.info("✓ Test dataset schema validation complete")

    logger.info(f"Data preparation complete: train={train_uri}, val={val_uri}, test={test_uri}")
    logger.info(f"Samples created: train_sample={train_sample_uri}, val_sample={val_sample_uri}")

    # Save checkpoint
    _save_phase_checkpoint(PhaseNames.PREPARE_DATA, context, on_checkpoint_saved)


# ============================================
# Phase 3: Baselines
# ============================================


def build_baselines(
    spark: SparkSession,
    context: BuildContext,
    config: Config,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
):
    """
    Phase 3: Build baseline models.

    Args:
        spark: SparkSession
        context: Build context
        config: Configuration
        on_checkpoint_saved: Optional callback for platform integration
    """

    logger.info("=== Phase 3: Baseline Models ===")

    baseline_agent = BaselineBuilderAgent(spark, context, config)

    # ============================================
    # Step 1: Build Baseline (with retry)
    # ============================================
    baseline = None
    max_attempts = 5
    error_history = []  # Track errors to pass to agent for learning

    for attempt in range(max_attempts):
        try:
            logger.info(f"Baseline generation attempt {attempt + 1}/{max_attempts}")

            # Pass error history to agent for debugging context
            if error_history:
                # Store errors in context for agent to access
                context.scratch["_baseline_errors"] = error_history
                logger.info(f"Previous errors: {len(error_history)} failures recorded")

            baseline = baseline_agent.run()
            break  # Success
        except Exception as e:
            error_msg = str(e)
            error_history.append({"attempt": attempt + 1, "error": error_msg, "error_type": type(e).__name__})
            logger.error(f"Baseline generation attempt {attempt + 1} failed: {error_msg}")

            if attempt < max_attempts - 1:
                logger.info(f"Retrying baseline generation... ({len(error_history)} previous failures)")
            else:
                logger.error("All baseline generation attempts failed")

    if baseline is None:
        raise RuntimeError(
            f"Failed to generate baseline after {max_attempts} attempts. Errors: {[e['error'] for e in error_history]}"
        )

    # ============================================
    # Step 2: Update Context and Save Report
    # ============================================
    context.update(heuristic_baseline=baseline)

    save_report(
        context.work_dir,
        "04_baseline",
        {
            "name": baseline.name,
            "performance": baseline.performance,
            "model_type": baseline.model_type,
            "metadata": baseline.metadata,
        },
    )

    logger.info(f"Baselines complete: {baseline.name} ({baseline.performance:.4f})")

    # Save checkpoint
    _save_phase_checkpoint(PhaseNames.BUILD_BASELINES, context, on_checkpoint_saved)


# ============================================
# Phase 4: Model Search
# ============================================


def _execute_variant(
    plan,
    solution_id: int,
    journal: SearchJournal,
    spark: SparkSession,
    config: Config,
    runner: TrainingRunner,
    pipelines_dir: Path,
    transformed_output_base: str,
    variant_context: BuildContext,
) -> Solution:
    """
    Execute a single variant plan (thread-safe).

    Args:
        variant_context: Isolated context with per-variant scratch space

    Returns:
        Solution object (successful or failed)
    """
    # Get parent node
    parent_node = (
        next((n for n in journal.nodes if n.solution_id == plan.parent_solution_id), None)
        if plan.parent_solution_id >= 0
        else None
    )

    # Validate parent exists if required
    if (
        not parent_node
        and plan.parent_solution_id >= 0
        and (plan.features.strategy == "reuse_parent" or plan.features.strategy == "modify_parent")
    ):
        logger.error(f"Plan requires parent solution {plan.parent_solution_id} but it doesn't exist")
        return Solution(
            solution_id=solution_id,
            feature_pipeline=None,  # type: ignore
            model=None,  # type: ignore
            model_type=plan.model.model_type,
            error=f"Parent solution {plan.parent_solution_id} not found",
            is_buggy=True,
            plan=plan,
            parent=None,
        )

    try:
        # Create/reuse feature pipeline
        if plan.features.strategy == "reuse_parent" and parent_node:
            # Deep copy to avoid race conditions when multiple threads reuse same parent
            feature_pipeline = copy.deepcopy(parent_node.feature_pipeline)
            parent_code_path = pipelines_dir / f"solution{plan.parent_solution_id}_pipeline.py"
            pipeline_code = parent_code_path.read_text() if parent_code_path.exists() else ""
        else:
            feature_pipeline, _ = FeatureProcessorAgent(
                spark=spark,
                train_uri=variant_context.train_sample_uri,
                context=variant_context,
                config=config,
                plan=plan.features,
            ).run()
            pipeline_code = variant_context.scratch.get("_saved_pipeline_code", "")  # Thread-safe: isolated scratch

        # Fit pipeline
        fitted_pipeline = fit_pipeline(
            dataset_uri=variant_context.train_sample_uri,
            pipeline=feature_pipeline,
            target_columns=variant_context.output_targets,
            group_column=variant_context.group_column,
        )

        # Inspect transformed schema
        sample_for_inspection = spark.read.parquet(variant_context.train_sample_uri).limit(100).toPandas()
        columns_to_drop_for_inspection = list(variant_context.output_targets)
        if variant_context.group_column and variant_context.group_column in sample_for_inspection.columns:
            columns_to_drop_for_inspection.append(variant_context.group_column)
        sample_features = sample_for_inspection.drop(columns=columns_to_drop_for_inspection, errors="ignore")
        sample_transformed_array = fitted_pipeline.transform(sample_features)
        num_output_features = sample_transformed_array.shape[1]
        transformed_schema = {
            "columns": [f"feature_{i}" for i in range(num_output_features)],
            "dtypes": {f"feature_{i}": "float64" for i in range(num_output_features)},
            "num_features": num_output_features,
        }

        # Create model
        model, _ = ModelDefinerAgent(
            model_type=plan.model.model_type,
            context=variant_context,
            config=config,
            transformed_schema=transformed_schema,
            plan=plan.model,
        ).run()

        # Capture Keras params (thread-safe: isolated scratch)
        solution_kwargs = {}
        if plan.model.model_type == "keras":
            solution_kwargs["optimizer"] = variant_context.scratch.get("_saved_optimizer")
            solution_kwargs["loss"] = variant_context.scratch.get("_saved_loss")
            solution_kwargs["epochs"] = variant_context.scratch.get("_keras_epochs", config.keras_default_epochs)
            solution_kwargs["batch_size"] = variant_context.scratch.get(
                "_keras_batch_size", config.keras_default_batch_size
            )

        # Create solution
        new_solution = Solution(
            solution_id=solution_id,
            feature_pipeline=feature_pipeline,
            model=model,
            model_type=plan.model.model_type,
            parent=parent_node,
            plan=plan,
            **solution_kwargs,
        )

        # NOTE: Parent-child linking happens after parallel execution completes (sequential, thread-safe)

        # Train and evaluate
        start_time = time.time()

        # Save pipeline code
        pipeline_code_path = pipelines_dir / f"solution{solution_id}_pipeline.py"
        pipeline_code_path.write_text(pipeline_code if pipeline_code else "# No code")

        # Transform samples (use integration-provided location)
        train_transformed_uri = f"{transformed_output_base}/solution{solution_id}_train.parquet"
        val_transformed_uri = f"{transformed_output_base}/solution{solution_id}_val.parquet"

        transform_dataset_via_spark(
            spark=spark,
            dataset_uri=variant_context.train_sample_uri,
            fitted_pipeline=fitted_pipeline,
            output_uri=train_transformed_uri,
            target_columns=variant_context.output_targets,
            pipeline_code=pipeline_code,
            group_column=variant_context.group_column,
        )

        transform_dataset_via_spark(
            spark=spark,
            dataset_uri=variant_context.val_sample_uri,
            fitted_pipeline=fitted_pipeline,
            output_uri=val_transformed_uri,
            target_columns=variant_context.output_targets,
            pipeline_code=pipeline_code,
            group_column=variant_context.group_column,
        )

        # Train
        training_kwargs = {}
        if plan.model.model_type == "keras":
            training_kwargs.update(
                {k: v for k, v in solution_kwargs.items() if k in ["optimizer", "loss", "epochs", "batch_size"]}
            )

        model_artifacts_path = runner.run_training(
            template=f"train_{plan.model.model_type}",
            model=model,
            feature_pipeline=fitted_pipeline,
            train_uri=train_transformed_uri,
            val_uri=val_transformed_uri,
            timeout=config.training_timeout,
            target_columns=variant_context.output_targets,
            group_column=variant_context.group_column,
            **training_kwargs,
        )

        # Save pipeline artifacts
        artifacts_dir = model_artifacts_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "pipeline.pkl", "wb") as f:
            cloudpickle.dump(fitted_pipeline, f)

        if pipeline_code:
            src_dir = model_artifacts_path / "src"
            src_dir.mkdir(exist_ok=True)
            (src_dir / "pipeline.py").write_text(pipeline_code)
            (src_dir / "__init__.py").write_text("# Auto-generated\n")

        # Evaluate
        performance = evaluate_on_sample(
            spark=spark,
            sample_uri=variant_context.val_sample_uri,
            model_artifacts_path=model_artifacts_path,
            model_type=plan.model.model_type,
            metric=variant_context.metric.name,
            target_columns=variant_context.output_targets,
            group_column=variant_context.group_column,
        )

        # Update solution
        new_solution.model_artifacts_path = model_artifacts_path
        new_solution.performance = float(performance)
        new_solution.training_time = time.time() - start_time
        new_solution.is_buggy = False

        logger.info(
            f"✓ Plan {plan.variant_id} (solution {solution_id}): {performance:.4f} ({new_solution.training_time:.1f}s)"
        )

        return new_solution

    except Exception as e:
        logger.error(f"Plan {plan.variant_id} (solution {solution_id}) failed: {e}")
        return Solution(
            solution_id=solution_id,
            feature_pipeline=None,  # type: ignore
            model=None,  # type: ignore
            model_type=plan.model.model_type,
            error=str(e),
            is_buggy=True,
            plan=plan,
            parent=parent_node,
        )


def search_models(
    spark: SparkSession,
    context: BuildContext,
    runner: TrainingRunner,
    search_policy: SearchPolicy,
    config: Config,
    integration: WorkflowIntegration,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
    restored_journal: SearchJournal | None = None,
    restored_insight_store: InsightStore | None = None,
) -> Solution | None:
    """
    Phase 4: Iterative tree-search for best model.

    Search operates on SAMPLES for fast iterations.
    Final model retrained on FULL dataset after search completes.

    Args:
        spark: SparkSession
        context: Build context
        runner: Training runner
        search_policy: Tree-search policy
        config: Configuration
        on_checkpoint_saved: Optional callback for platform integration
        restored_journal: Restored SearchJournal from checkpoint (for resume)
        restored_insight_store: Restored InsightStore from checkpoint (for resume)

    Returns:
        Best solution (trained on full dataset)
    """

    logger.info("=== Phase 4: Model Search (Hypothesis-Driven on Samples) ===")

    # ============================================
    # Step 1: Initialize Search Infrastructure
    # ============================================
    # Use restored journal/insight_store if resuming, otherwise create fresh
    if restored_journal:
        journal = restored_journal
        logger.info(f"Using restored SearchJournal with {len(journal.nodes)} existing solutions")
    else:
        journal = SearchJournal(baseline=context.heuristic_baseline)

    if restored_insight_store:
        insight_store = restored_insight_store
        context.insight_store = insight_store
        logger.info(f"Using restored InsightStore with {len(insight_store)} existing insights")
    else:
        insight_store = InsightStore()
        context.insight_store = insight_store
        logger.info("Insight store initialized")

    # Prepare directories (transformed location from integration, pipelines always local)
    transformed_output_base = integration.get_artifact_location(
        "transformed", context.dataset_uri, context.experiment_id, context.work_dir
    )

    # For local paths, mkdir; for S3, no-op (Spark handles creation)
    if not transformed_output_base.startswith("s3://"):
        Path(transformed_output_base).mkdir(parents=True, exist_ok=True)

    pipelines_dir = context.work_dir / DirNames.BUILD_DIR / "search" / "pipelines"
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    # Track solution IDs independently from iteration number
    # When resuming from checkpoint, start from max existing solution_id + 1
    if restored_journal and restored_journal.nodes:
        solution_id_counter = max(s.solution_id for s in restored_journal.nodes) + 1
        logger.info(f"Resuming solution_id_counter from {solution_id_counter}")
    else:
        solution_id_counter = 0

    # ============================================
    # Step 2: HYPOTHESIS-DRIVEN SEARCH LOOP
    # ============================================
    # NOTE: max_search_iterations = number of hypothesis rounds
    # Each round creates multiple variants (typically 3), so total solutions = iterations × variants_per_round

    for iteration in range(config.max_search_iterations):
        with tracer.start_as_current_span(f"Iteration {iteration}"):
            logger.info(f"=== Iteration {iteration + 1}/{config.max_search_iterations} ===")

            # ============================================
            # Step 2a: Policy Picks Node to Expand
            # ============================================
            parent_node = search_policy.decide_next_solution(journal, context, iteration, config.max_search_iterations)
            expand_solution_id = parent_node.solution_id if parent_node else None

            logger.info(
                f"Expanding solution {expand_solution_id if expand_solution_id is not None else 'from scratch'}"
            )

            # ============================================
            # Step 2b/2c: Generate Plans (with or without hypothesis)
            # ============================================
            if expand_solution_id is None:
                # Bootstrap mode: Create diverse initial solutions without hypothesis
                logger.info("Bootstrap mode: creating diverse initial solution from scratch")
                try:
                    plans = PlannerAgent(
                        journal=journal,
                        context=context,
                        config=config,
                        hypothesis=None,  # Bootstrap mode
                        num_bootstrap=3,  # One diverse solution per round
                    ).run()

                    logger.info(f"Generated {len(plans)} bootstrap plan(s)")

                except Exception as e:
                    logger.error(f"Bootstrap planning failed: {e}")
                    continue  # Skip this iteration

            else:
                # Hypothesis-driven mode: Generate hypothesis then plans
                try:
                    hypothesis = HypothesiserAgent(
                        journal=journal,
                        context=context,
                        config=config,
                        expand_solution_id=expand_solution_id,
                    ).run()

                    logger.info(
                        f"Hypothesis: {hypothesis.focus} - vary '{hypothesis.vary}', {hypothesis.num_variants} variants"
                    )

                except Exception as e:
                    logger.error(f"Hypothesis generation failed: {e}")
                    # Fallback: create simple improvement hypothesis
                    hypothesis = Hypothesis(
                        expand_solution_id=expand_solution_id,
                        focus="both",
                        vary="general_improvement",
                        num_variants=1,
                        rationale="Fallback - hypothesis generation failed",
                        keep_from_parent=[],
                        expected_impact="unknown",
                    )

                # Generate plans from hypothesis
                try:
                    plans = PlannerAgent(
                        journal=journal,
                        context=context,
                        config=config,
                        hypothesis=hypothesis,
                    ).run()

                    logger.info(f"Generated {len(plans)} plan variants")

                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    continue  # Skip this iteration

            # ============================================
            # Step 2d: Execute Plan Variants (in parallel)
            # ============================================
            # Pre-assign solution IDs to avoid race conditions
            variant_ids = list(range(solution_id_counter, solution_id_counter + len(plans)))

            # Execute variants in parallel
            logger.info(f"Executing {len(plans)} variants in parallel (max_workers={config.max_parallel_variants})...")
            with ThreadPoolExecutor(max_workers=config.max_parallel_variants) as executor:
                futures = []
                for plan, variant_id in zip(plans, variant_ids):
                    # Create isolated context for each variant (shallow copy with new scratch)
                    # NOTE: Only scratch is isolated per-variant; all other attributes are shared (read-only)
                    variant_context = copy.copy(context)
                    variant_context.scratch = {}  # Fresh scratch dict per variant

                    futures.append(
                        executor.submit(
                            _execute_variant,
                            plan,
                            variant_id,
                            journal,
                            spark,
                            config,
                            runner,
                            pipelines_dir,
                            transformed_output_base,
                            variant_context,
                        )
                    )

                # Collect results with error handling
                variant_solutions = []
                for f in futures:
                    try:
                        variant_solutions.append(f.result())
                    except Exception as e:
                        logger.error(f"Variant execution thread failed unexpectedly: {e}", exc_info=True)
                        # Note: _execute_variant() already catches exceptions and returns failed Solution objects
                        # This catches any unforeseen thread-level failures

            # Link parent-child relationships and add to journal (sequential, thread-safe)
            for sol in variant_solutions:
                if sol.parent:
                    sol.parent.children.append(sol)
                journal.add_node(sol)

            # Update counter based on pre-assigned IDs to prevent collisions
            # This may create gaps if threads fail, but gaps are acceptable (collisions are not)
            solution_id_counter += len(variant_ids)

            # ============================================
            # Step 2e: Extract Insights from Variants (skip in bootstrap mode)
            # ============================================
            if variant_solutions and expand_solution_id is not None:
                # Only extract insights when we have a hypothesis to learn from
                try:
                    InsightExtractorAgent(
                        hypothesis=hypothesis,
                        variant_solutions=variant_solutions,
                        insight_store=insight_store,
                        context=context,
                        config=config,
                    ).run()
                except Exception as e:
                    logger.warning(f"Insight extraction failed: {e}")

            # ============================================
            # Step 2f: Check Stopping
            # ============================================
            if search_policy.should_stop(journal, iteration, config.max_search_iterations):
                logger.info("Search policy triggered early stop")
                break

    # ============================================
    # Step 5: Retrain Best Solution on FULL Dataset
    # ============================================
    best_solution = journal.best_node

    if not best_solution:
        logger.warning("No successful solutions found during search")
        return None

    logger.info(
        f"Search complete! Best solution: solution_id={best_solution.solution_id}, perf={best_solution.performance:.4f}"
    )
    logger.info(journal.summarize())

    # Retrain on full data
    final_solution = retrain_on_full_dataset(
        spark=spark, best_solution=best_solution, context=context, runner=runner, config=config
    )

    # Save checkpoint with journal and insight_store
    _save_phase_checkpoint(
        PhaseNames.SEARCH_MODELS, context, on_checkpoint_saved, search_journal=journal, insight_store=insight_store
    )

    # Store journal in context for fallback logic access
    context.scratch["_search_journal"] = journal

    return final_solution


# ============================================
# Final Retraining on Full Dataset
# ============================================


def retrain_on_full_dataset(
    spark: SparkSession, best_solution: Solution, context: BuildContext, runner: TrainingRunner, config: Config
) -> Solution:
    """
    Retrain best solution on FULL dataset.

    Search was performed on samples (fast iterations).
    Now retrain with same configuration on full data (slow but accurate).

    Args:
        spark: SparkSession
        best_solution: Best solution from tree-search
        context: Build context
        runner: Training runner
        config: Configuration

    Returns:
        Final solution trained on full dataset
    """

    logger.info("=== Retraining Best Solution on Full Dataset ===")
    logger.info(f"Best solution: solution_id={best_solution.solution_id}, sample perf={best_solution.performance:.4f}")

    # ============================================
    # Step 1: Fit Pipeline on Sample (Not Full Dataset)
    # ============================================
    # LIMITATION: Fitting final pipeline on sample instead of full dataset to avoid OOM.
    # When datasets are in S3, we cannot load 10M+ rows into pandas for fitting.
    # We fit on the same sample used during search (~30k rows), which works because:
    # - StandardScaler: mean/std from 30k ≈ mean/std from 10M (law of large numbers)
    # - OneHotEncoder: captures categories if sample is representative
    # - Other transformers: similarly stable on large samples
    # However, this MAY reduce quality if sample is not perfectly representative.
    # TODO: Implement Spark-based fitting or use Dask for true out-of-core processing.
    logger.info("Fitting pipeline on training sample (same as search)...")

    fitted_pipeline = fit_pipeline(
        dataset_uri=context.train_sample_uri,  # ← SAMPLE (not full dataset)
        pipeline=best_solution.feature_pipeline,
        target_columns=context.output_targets,
        group_column=context.group_column,
    )

    # ============================================
    # Step 2: Transform FULL Datasets via Spark
    # ============================================
    logger.info("Transforming full datasets via Spark (may take 10-30 minutes for large data)...")

    # Extract pipeline code from disk (best solution's artifacts)
    pipeline_code_path = best_solution.model_artifacts_path / "src" / "pipeline.py"
    pipeline_code = pipeline_code_path.read_text() if pipeline_code_path.exists() else ""

    final_train_transformed = str(context.work_dir / DirNames.BUILD_DIR / "data" / "final_transformed" / "train")
    final_val_transformed = str(context.work_dir / DirNames.BUILD_DIR / "data" / "final_transformed" / "val")

    transform_dataset_via_spark(
        spark=spark,
        dataset_uri=context.train_uri,  # ← FULL training data
        fitted_pipeline=fitted_pipeline,
        output_uri=final_train_transformed,
        target_columns=context.output_targets,
        pipeline_code=pipeline_code,
        group_column=context.group_column,
    )

    transform_dataset_via_spark(
        spark=spark,
        dataset_uri=context.val_uri,  # ← FULL validation data
        fitted_pipeline=fitted_pipeline,
        output_uri=final_val_transformed,
        target_columns=context.output_targets,
        pipeline_code=pipeline_code,
        group_column=context.group_column,
    )

    logger.info("Transformation complete!")

    # ============================================
    # Step 3: Train Final Model on FULL Data
    # ============================================
    logger.info("Training final model on full dataset (may take 10-60 minutes)...")

    # For Keras, pass optimizer, loss, and training params
    retrain_kwargs = {}
    if best_solution.model_type == "keras":
        retrain_kwargs["optimizer"] = best_solution.optimizer
        retrain_kwargs["loss"] = best_solution.loss
        retrain_kwargs["epochs"] = best_solution.epochs
        retrain_kwargs["batch_size"] = best_solution.batch_size

    final_artifacts_path = runner.run_training(
        template=f"train_{best_solution.model_type}",
        model=best_solution.model,  # ← Untrained model object
        feature_pipeline=fitted_pipeline,
        train_uri=final_train_transformed,  # ← FULL transformed data
        val_uri=final_val_transformed,
        timeout=config.training_timeout,
        target_columns=context.output_targets,
        group_column=context.group_column,
        **retrain_kwargs,
    )

    # Save pipeline to artifacts/ (aligned structure)
    artifacts_dir = final_artifacts_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    pipeline_path = artifacts_dir / "pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        cloudpickle.dump(fitted_pipeline, f)
    logger.debug(f"Saved final pipeline to {pipeline_path}")

    # Save pipeline code to src/ (aligned structure)
    if pipeline_code:
        src_dir = final_artifacts_path / "src"
        src_dir.mkdir(exist_ok=True)
        code_path = src_dir / "pipeline.py"
        code_path.write_text(pipeline_code)
        # Create __init__.py
        (src_dir / "__init__.py").write_text("# Auto-generated\n")
        logger.debug(f"Saved pipeline code to {code_path}")

    logger.info(f"Final training complete! Artifacts saved to: {final_artifacts_path}")

    # ============================================
    # Step 4: Evaluate Final Model on Full Validation Set
    # ============================================
    logger.info("Evaluating final model on full validation set...")

    final_val_performance = evaluate_on_sample(
        spark=spark,
        sample_uri=context.val_uri,  # ← FULL validation set
        model_artifacts_path=final_artifacts_path,
        model_type=best_solution.model_type,
        metric=context.metric.name,
        target_columns=context.output_targets,
        group_column=context.group_column,
    )

    logger.info(f"Final validation performance: {final_val_performance:.4f}")
    logger.info(
        f"  Improvement over sample: {best_solution.performance:.4f} → {final_val_performance:.4f} "
        f"({(final_val_performance - best_solution.performance):.4f})"
    )

    # ============================================
    # Step 5: Return Final Solution
    # ============================================
    final_solution_kwargs = {}
    if best_solution.model_type == "keras":
        final_solution_kwargs["optimizer"] = best_solution.optimizer
        final_solution_kwargs["loss"] = best_solution.loss
        final_solution_kwargs["epochs"] = best_solution.epochs
        final_solution_kwargs["batch_size"] = best_solution.batch_size

    return Solution(
        solution_id=best_solution.solution_id,
        feature_pipeline=fitted_pipeline,
        model=best_solution.model,  # Same model instance (now trained)
        model_type=best_solution.model_type,
        model_artifacts_path=final_artifacts_path,
        performance=final_val_performance,  # Full validation performance
        parent=best_solution.parent,
        training_time=best_solution.training_time,
        **final_solution_kwargs,
    )


# ============================================
# Phase 5: Final Evaluation
# ============================================


def evaluate_final(
    spark: SparkSession,
    context: BuildContext,
    solution: Solution,
    config: Config,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
) -> dict:
    """
    Phase 5: Final evaluation on test set sample.

    Args:
        spark: SparkSession
        context: Build context
        solution: Best solution
        config: Configuration
        on_checkpoint_saved: Optional callback for platform integration

    Returns:
        Final test metrics dict (for backward compatibility) or None if evaluation fails
    """
    logger.info("=== Phase 5: Final Evaluation ===")

    # Step 1: Load test sample to pandas
    if context.test_uri:
        logger.info(f"Loading test sample from {context.test_uri}")
        test_df_spark = spark.read.parquet(context.test_uri)
        # Sample for evaluation (20k-50k rows)
        sample_size = min(50000, test_df_spark.count())
        test_sample_df = test_df_spark.limit(sample_size).toPandas()
        logger.info(f"Loaded {len(test_sample_df)} test samples for evaluation")
    else:
        logger.warning("No test set available - using validation set for evaluation")
        val_df_spark = spark.read.parquet(context.val_uri)
        sample_size = min(50000, val_df_spark.count())
        test_sample_df = val_df_spark.limit(sample_size).toPandas()
        logger.info(f"Loaded {len(test_sample_df)} validation samples for evaluation")

    # Step 2: Load predictor from solution artifacts
    if not solution.model_artifacts_path or not solution.model_artifacts_path.exists():
        logger.error("Solution model artifacts path does not exist - cannot evaluate")
        return None

    # Load predictor class dynamically
    predictor_file = solution.model_artifacts_path / "predictor.py"
    if not predictor_file.exists():
        # Fallback: try copying predictor template first
        logger.warning("predictor.py not found in artifacts - attempting to create from template")
        template_file = f"{solution.model_type}_predictor.py"
        template_path = Path(__file__).parent / "templates" / "inference" / template_file

        if template_path.exists():
            shutil.copy2(template_path, predictor_file)
            logger.info(f"Created predictor from template: {template_file}")
        else:
            logger.error(f"Predictor template not found: {template_path}")
            return None

    # Import predictor module
    try:
        spec = importlib.util.spec_from_file_location("predictor_module", predictor_file)
        predictor_module = importlib.util.module_from_spec(spec)
        sys.modules["predictor_module"] = predictor_module
        spec.loader.exec_module(predictor_module)

        # Get predictor class (XGBoostPredictor, CatBoostPredictor, or KerasPredictor)
        if solution.model_type == "xgboost":
            predictor_class = predictor_module.XGBoostPredictor
        elif solution.model_type == "catboost":
            predictor_class = predictor_module.CatBoostPredictor
        elif solution.model_type == "keras":
            predictor_class = predictor_module.KerasPredictor
        else:
            logger.error(f"Unknown model type: {solution.model_type}")
            return None

        # Instantiate predictor
        predictor = predictor_class(str(solution.model_artifacts_path))
        logger.info(f"Loaded {solution.model_type} predictor from {solution.model_artifacts_path}")

    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        return None

    # Step 3: Run ModelEvaluatorAgent
    agent = ModelEvaluatorAgent(spark=spark, context=context, config=config)

    try:
        evaluation_report = agent.run(
            solution=solution,
            test_sample_df=test_sample_df,
            predictor=predictor,
        )

        if evaluation_report is None:
            logger.error("Model evaluation failed - agent returned None")
            return None

        # Save evaluation report (convert to dict for clean YAML serialization)
        save_report(context.work_dir, "05_final_evaluation", evaluation_report.to_dict())

        # Extract metrics dict for backward compatibility
        metrics = {
            "verdict": evaluation_report.verdict,
            "deployment_ready": evaluation_report.deployment_ready,
            "performance": (
                evaluation_report.core_metrics.primary_metric_value if evaluation_report.core_metrics else None
            ),
            "all_metrics": evaluation_report.core_metrics.all_metrics if evaluation_report.core_metrics else {},
            "summary": evaluation_report.summary,
        }

        logger.info(f"Final evaluation complete: verdict={evaluation_report.verdict}")

        # Store evaluation in scratch for packaging phase
        context.scratch["_evaluation_report"] = evaluation_report

        # Save checkpoint (with insight_store if available)
        insight_store = context.insight_store if hasattr(context, "insight_store") else None
        _save_phase_checkpoint(PhaseNames.EVALUATE_FINAL, context, on_checkpoint_saved, insight_store=insight_store)

        return metrics

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        return None


# ============================================
# Phase 6: Package Final Deliverables
# ============================================


def package_final_model(
    spark: SparkSession,
    context: BuildContext,
    solution: Solution,
    final_metrics: dict,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None = None,
) -> Path:
    """
    Package all final deliverables into a unified directory.

    Creates a final model package containing:
    - Model artifacts (model.pkl, pipeline.pkl)
    - Predictor code (inference template)
    - Reports (stats, task analysis, evaluation)
    - Metadata

    Args:
        spark: SparkSession for loading training data to generate schemas
        context: Build context with all workflow state
        solution: Final trained solution
        final_metrics: Test evaluation metrics
        on_checkpoint_saved: Optional callback for platform integration

    Returns:
        Path to final package directory
    """
    logger.info("=== Phase 6: Packaging Final Deliverables ===")

    # Create final package directory
    package_dir = context.work_dir / "model"
    package_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Step 1: Copy Model Artifacts Tree (already in final structure)
    # ============================================
    if solution.model_artifacts_path:
        logger.info(f"Copying model artifacts tree from {solution.model_artifacts_path}...")

        # Copy entire tree (artifacts/ and src/ already exist from training/search)
        for item in solution.model_artifacts_path.iterdir():
            if item.is_dir():
                # Copy entire directories (artifacts/, src/)
                shutil.copytree(item, package_dir / item.name, dirs_exist_ok=True)
                logger.info(f"  Copied directory: {item.name}/")
            else:
                # Copy any root-level files
                shutil.copy2(item, package_dir / item.name)
                logger.info(f"  Copied file: {item.name}")

    # ============================================
    # Step 2: Add Predictor Template
    # ============================================
    logger.info("Adding predictor code...")

    if solution.model_type == "baseline":
        # Special case: baseline fallback - copy baseline predictor as predictor.py
        baseline = getattr(context, "heuristic_baseline", None)
        if baseline is None:
            logger.error("Baseline solution requested, but context.heuristic_baseline is None.")
            raise ValueError("Baseline predictor packaging failed: no heuristic baseline available in context.")

        baseline_name = baseline.metadata.get("predictor_code_file", f"{baseline.name}.py")
        baseline_source = solution.model_artifacts_path / baseline_name

        if baseline_source.exists():
            shutil.copy2(baseline_source, package_dir / "predictor.py")
            logger.info(f"  Copied baseline predictor: {baseline_name} → predictor.py")
        else:
            logger.error(f"Baseline predictor not found: {baseline_source}")
            raise FileNotFoundError(f"Baseline predictor file missing: {baseline_source}")
    else:
        # Normal case: XGBoost/Keras templates
        template_file = f"{solution.model_type}_predictor.py"
        template_path = Path(__file__).parent / "templates" / "inference" / template_file

        if template_path.exists():
            shutil.copy2(template_path, package_dir / "predictor.py")
            logger.info(f"  Copied predictor template: {template_file}")
        else:
            logger.warning(f"Predictor template not found: {template_path}")

    # ============================================
    # Step 2b: Add Training Code for Retraining Support
    # ============================================
    logger.info("Adding training code for retraining support...")

    # Get training template based on solution's model type
    training_template = Path(__file__).parent / "templates" / "training" / f"train_{solution.model_type}.py"

    if training_template.exists():
        # Ensure src/ directory exists
        src_dir = package_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Copy training template as trainer.py
        shutil.copy2(training_template, src_dir / "trainer.py")
        logger.info(f"  Copied training template: train_{solution.model_type}.py → src/trainer.py")
    else:
        logger.warning(f"Training template not found: {training_template}")

    # ============================================
    # Step 3: Generate Input/Output Schemas from Training Data
    # ============================================
    logger.info("Generating input/output schemas from training data...")

    # Load small sample to infer actual types
    train_sample = spark.read.parquet(context.train_sample_uri).limit(100).toPandas()

    # Input schema: actual feature columns with real types
    feature_cols = [col for col in train_sample.columns if col not in context.output_targets]
    X_sample = train_sample[feature_cols]

    input_schema = {
        "type": "object",
        "properties": {
            col: {
                "type": _pandas_dtype_to_json_schema_type(X_sample[col].dtype),
                "description": f"Feature: {col}",
            }
            for col in feature_cols
        },
        "required": list(feature_cols),
        "title": "ModelInput",
    }

    # Output schema: predictor always returns {"prediction": ...}
    y_sample = train_sample[context.output_targets[0]]
    output_type = _pandas_dtype_to_json_schema_type(y_sample.dtype)

    output_schema = {
        "type": "object",
        "properties": {"prediction": {"type": output_type, "description": f"Predicted {context.output_targets[0]}"}},
        "required": ["prediction"],
        "title": "ModelOutput",
    }

    logger.info(f"  Input schema: {len(feature_cols)} features")
    logger.info(f"  Output schema: {output_type} prediction")

    # Save all schemas to schemas/ directory
    schemas_dir = package_dir / "schemas"
    schemas_dir.mkdir(exist_ok=True)

    # Save external API schemas (JSON Schema format)
    with open(schemas_dir / "input.json", "w") as f:
        json.dump(input_schema, f, indent=2)
    with open(schemas_dir / "output.json", "w") as f:
        json.dump(output_schema, f, indent=2)

    # Save internal pandas dtypes for predictor type conversion
    dtypes_dict = {col: str(dtype) for col, dtype in X_sample.dtypes.items()}
    with open(schemas_dir / "dtypes.json", "w") as f:
        json.dump(dtypes_dict, f, indent=2)

    logger.info(f"  Saved schemas: input.json, output.json, dtypes.json ({len(dtypes_dict)} columns)")

    # ============================================
    # Step 4: Save Analysis Reports to evaluation/ subdirectory
    # ============================================
    logger.info("Saving analysis reports...")

    # Create evaluation directory and reports subdirectory
    evaluation_dir = package_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    reports_subdir = evaluation_dir / "reports"
    reports_subdir.mkdir(exist_ok=True)

    # Save metrics summary (includes all_metrics for deployment script compatibility)
    metrics_summary = {
        "metric": context.metric.name,
        "value": float(final_metrics.get("performance", 0)),
        "optimization_direction": context.metric.optimization_direction,
        "baseline": float(context.heuristic_baseline.performance) if context.heuristic_baseline else None,
        "test_samples": final_metrics.get("test_samples", 0),
        "all_metrics": final_metrics.get("all_metrics", {}),
    }

    with open(evaluation_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("  Saved metrics summary to: evaluation/metrics.json")

    # Save detailed reports to separate files
    with open(reports_subdir / "statistical.json", "w") as f:
        json.dump(context.stats, f, indent=2, default=str)

    with open(reports_subdir / "task_analysis.json", "w") as f:
        json.dump(context.task_analysis, f, indent=2, default=str)

    # Save insights if available
    insights_data = [i.to_dict() for i in context.insight_store.get_all()] if context.insight_store else []
    if insights_data:
        with open(reports_subdir / "insights.json", "w") as f:
            json.dump(insights_data, f, indent=2, default=str)

    # Save evaluation report if available
    evaluation_report = context.scratch.get("_evaluation_report")
    if evaluation_report is not None:
        with open(reports_subdir / "evaluation.json", "w") as f:
            json.dump(evaluation_report.to_dict(), f, indent=2, default=str)

    logger.info("  Saved detailed reports to: evaluation/reports/{statistical,task_analysis,insights,evaluation}.json")

    # ============================================
    # Step 4: Create README from Template
    # ============================================
    logger.info("Creating README...")

    # Load README template
    template_path = Path(__file__).parent / "templates" / "packaging" / "README.md.template"
    with open(template_path) as f:
        readme_template = f.read()

    # Render template with context
    predictor_class_map = {
        "xgboost": "XGBoostPredictor",
        "catboost": "CatBoostPredictor",
        "keras": "KerasPredictor",
    }
    predictor_class = predictor_class_map.get(solution.model_type, f"{solution.model_type.capitalize()}Predictor")
    readme_content = readme_template.format(
        experiment_id=context.experiment_id,
        intent=context.intent,
        model_type=solution.model_type,
        task_type=context.task_analysis.get("task_type", "unknown"),
        target_columns=", ".join(context.output_targets),
        metric_name=context.metric.name,
        metric_direction=context.metric.optimization_direction,
        baseline_performance=f"{context.heuristic_baseline.performance:.4f}" if context.heuristic_baseline else "N/A",
        final_performance=(
            f"{final_metrics.get('performance', 'N/A'):.4f}"
            if isinstance(final_metrics.get("performance"), int | float)
            else "N/A"
        ),
        test_samples=final_metrics.get("test_samples", "N/A"),
        predictor_class=predictor_class,
    )

    # Add fallback warning if applicable
    if final_metrics.get("fallback_used"):
        fallback_warning = f"""

## ⚠️ FALLBACK MODEL NOTICE

This is a **fallback model**. The highest-performing model from search was rejected during evaluation.

- **Original Model**: Solution {final_metrics.get('original_solution_id', 'unknown')} (rejected due to validation failures)
- **This Model**: Solution {final_metrics.get('fallback_solution_id', 'unknown')}
- **Reason for Fallback**: {final_metrics.get('fallback_reason', 'Primary model failed validation')}

The packaged model passed validation checks and is safe for deployment.
Refer to `evaluation/reports/evaluation.json` for detailed analysis.

"""
        readme_content += fallback_warning

    readme_path = package_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    logger.info("  Created README.md")

    # ============================================
    # Step 4a: Create model.yaml with Core Metadata
    # ============================================
    logger.info("Creating model.yaml with core metadata...")

    # Load training metadata to extract hyperparameters
    training_metadata_path = package_dir / "artifacts" / "metadata.json"
    if training_metadata_path.exists():
        with open(training_metadata_path) as f:
            training_metadata = json.load(f)
    else:
        training_metadata = {}

    # Create config directory and save hyperparameters
    config_dir = package_dir / "config"
    config_dir.mkdir(exist_ok=True)

    if "hyperparameters" in training_metadata:
        with open(config_dir / "hyperparameters.json", "w") as f:
            json.dump(training_metadata["hyperparameters"], f, indent=2)
        logger.info("  Saved hyperparameters to: config/hyperparameters.json")

    # Create clean model.yaml (MLflow-inspired)
    model_metadata = {
        "model_format": "plexe_v1",
        "intent": context.intent,
        "model_type": solution.model_type,
        "task_type": context.task_analysis.get("task_type", "unknown"),
        "target_column": context.output_targets[0] if context.output_targets else None,
        "output_targets": context.output_targets,
        "metric": {
            "name": context.metric.name,
            "value": float(final_metrics.get("performance", 0)),
            "optimization_direction": context.metric.optimization_direction,
            "baseline": float(context.heuristic_baseline.performance) if context.heuristic_baseline else None,
        },
        "training": {
            "features_count": training_metadata.get("n_features", len(feature_cols)),
            "train_samples": training_metadata.get("train_samples"),
            "val_samples": training_metadata.get("val_samples"),
        },
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": context.experiment_id,
            "user_id": context.user_id,
            "trained_by": "plexe",
        },
    }

    # Add fallback information if applicable
    if final_metrics.get("fallback_used"):
        model_metadata["fallback_info"] = {
            "is_fallback": True,
            "reason": final_metrics.get("fallback_reason", "Primary model failed validation"),
            "original_solution_id": final_metrics.get("original_solution_id"),
            "fallback_solution_id": final_metrics.get("fallback_solution_id"),
            "fallback_performance": float(final_metrics.get("performance", 0)),
        }

    with open(package_dir / "model.yaml", "w") as f:
        yaml.dump(model_metadata, f, default_flow_style=False, sort_keys=False)
    logger.info("  Created model.yaml")

    # Note: Keep artifacts/metadata.json for debugging (contains full training metadata)
    # Hyperparameters are duplicated in config/ for convenience

    # ============================================
    # Step 5: Create Tarball Archive
    # ============================================
    logger.info("Creating tarball archive...")

    tarball_path = context.work_dir / "model.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        # Add each item from package_dir at root level (not wrapped in model/)
        for item in package_dir.iterdir():
            tar.add(item, arcname=item.name)

    logger.info(f"  Created tarball: {tarball_path}")
    logger.info(f"  Tarball size: {tarball_path.stat().st_size / (1024**2):.2f} MB")

    logger.info("Packaging complete! Final deliverables:")
    logger.info(f"  Directory: {package_dir}")
    logger.info(f"  Archive: {tarball_path}")

    # Save checkpoint (with insight_store if available)
    insight_store = context.insight_store if hasattr(context, "insight_store") else None
    _save_phase_checkpoint(PhaseNames.PACKAGE_FINAL_MODEL, context, on_checkpoint_saved, insight_store=insight_store)

    return package_dir


# ============================================
# Helper Functions (Imported)
# ============================================


def _baseline_to_solution(baseline: Baseline, work_dir: Path) -> Solution:
    """Convert baseline to solution structure for consistency."""
    return Solution(
        solution_id=-1,
        feature_pipeline=None,  # type: ignore
        model=None,  # type: ignore - baseline doesn't have a model object
        model_type=baseline.model_type,
        model_artifacts_path=baseline.model_artifacts_path or work_dir / DirNames.BUILD_DIR / "search" / "baselines" / baseline.name,  # type: ignore
        performance=baseline.performance,
    )


def _pandas_dtype_to_json_schema_type(dtype) -> str:
    """
    Map pandas dtype to JSON Schema type using pandas API.

    Args:
        dtype: Pandas dtype object

    Returns:
        JSON Schema type string (boolean, integer, number, string)
    """
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_integer_dtype(dtype):
        return "integer"
    elif pd.api.types.is_float_dtype(dtype):
        return "number"
    else:
        # datetime, string, object, categorical, etc.
        return "string"


def _save_phase_checkpoint(
    phase_name: str,
    context: BuildContext,
    on_checkpoint_saved: Callable[[str, Path, Path], None] | None,
    search_journal: SearchJournal | None = None,
    insight_store: InsightStore | None = None,
) -> None:
    """
    Save checkpoint for a completed phase.

    Args:
        phase_name: Phase name (e.g., "analyze_data", "prepare_data")
        context: BuildContext with workflow state
        on_checkpoint_saved: Optional callback(phase_name, checkpoint_path, work_dir) for external sync
        search_journal: SearchJournal (only for Phase 4+)
        insight_store: InsightStore (only for Phase 4+)
    """
    try:
        # Save checkpoint to local disk
        checkpoint_path = save_checkpoint(
            experiment_id=context.experiment_id,
            phase_name=phase_name,
            context=context,
            work_dir=context.work_dir,
            search_journal=search_journal,
            insight_store=insight_store,
        )

        if checkpoint_path:
            logger.info(f"✓ Checkpoint saved: {phase_name}")

            # Invoke integration callback for external sync (S3, DynamoDB, etc.)
            if on_checkpoint_saved:
                try:
                    on_checkpoint_saved(phase_name, checkpoint_path, context.work_dir)
                except Exception as e:
                    logger.warning(f"External sync failed (non-critical): {e}")
        else:
            logger.warning(f"Checkpoint save failed for {phase_name} (workflow continues)")

    except Exception as e:
        logger.warning(f"Checkpoint save failed for {phase_name}: {e} (workflow continues)")
