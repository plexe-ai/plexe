"""
Model Evaluator Agent for comprehensive ML model evaluation.

Performs multi-phase evaluation through focused coding tasks that compose into a complete EvaluationReport.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import pandas as pd
from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.models import BuildContext, Solution, EvaluationReport
from plexe.config import Config
from plexe.tools.submission import (
    get_register_core_metrics_tool,
    get_register_diagnostic_report_tool,
    get_register_robustness_report_tool,
    get_register_explainability_report_tool,
    get_register_baseline_comparison_tool,
    get_register_final_evaluation_tool,
)
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class ModelEvaluatorAgent:
    """
    Agent that performs comprehensive model evaluation through structured phases.

    Each phase produces a component report that composes into the final EvaluationReport.

    TODO: Core Metrics phase could be moved to Spark for large-scale evaluation on full test sets.
          Currently uses pandas samples (20k-50k rows) which is sufficient for most cases but may
          not provide accurate metrics on 100GB+ datasets. Spark implementation would allow computing
          exact metrics on arbitrarily large test sets.

    TODO: Baseline Comparison could use Spark-computed metrics from Core Metrics phase for
          more accurate comparisons on large datasets. Currently both baseline and model metrics
          are computed on pandas samples.
    """

    def __init__(
        self,
        spark: SparkSession,
        context: BuildContext,
        config: Config,
    ):
        """
        Initialize the model evaluator agent.

        Args:
            spark: SparkSession for loading test data
            context: Build context
            config: Configuration
        """
        self.spark = spark
        self.context = context
        self.config = config
        self.llm_model = config.evaluation_llm

    def _build_agent(self, phase_name: str, phase_prompt: str, tools: list) -> CodeAgent:
        """
        Build CodeAgent for a specific evaluation phase.

        Args:
            phase_name: Name of the phase (for logging)
            phase_prompt: Prompt for this phase
            tools: List of tools for this phase

        Returns:
            Configured CodeAgent
        """
        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name=f"ModelEvaluator_{phase_name}",
            instructions=(
                "## YOUR ROLE:\\n"
                "You are an expert ML model evaluator performing systematic, rigorous evaluation.\\n\\n"
                "## INJECTED OBJECTS:\\n"
                "These objects are directly available to you:\\n"
                "- solution: Solution object with trained model artifacts\\n"
                "- test_sample_df: pandas DataFrame with test data sample\\n"
                "- predictor: Trained predictor instance (XGBoostPredictor or KerasPredictor)\\n"
                "- primary_metric_name: Name of the primary optimization metric (string)\\n"
                "- output_targets: list[str] (target column names to exclude from features)\\n\\n"
                "PREDICTOR INTERFACE:\\n"
                "The predictor's predict() function takes a pandas DataFrame (features only, no target)\\n"
                "and returns a pandas DataFrame with a 'prediction' column.\\n\\n"
                "Example usage:\\n"
                "```python\\n"
                "# Prepare features (drop target columns using output_targets)\\n"
                "feature_cols = [col for col in test_sample_df.columns if col not in output_targets]\\n"
                "X_test = test_sample_df[feature_cols]\\n"
                "y_true = test_sample_df[output_targets[0]].values\\n\\n"
                "# Generate predictions (returns DataFrame with 'prediction' column)\\n"
                "predictions_df = predictor.predict(X_test)\\n"
                "y_pred = predictions_df['prediction'].values\\n"
                "```\\n\\n"
                "## YOUR MISSION:\\n"
                f"{phase_prompt}\\n\\n"
                "CRITICAL: Always register your results using the specified tool.\\n"
                "CRITICAL: Provide interpretation, not just numbers.\\n"
                "IMPORTANT: Do not create plots or visualizations (headless environment).\\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=tools,
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["collections", "pandas", "pandas.*", "numpy", "numpy.*", "sklearn", "sklearn.*"],
            max_steps=20,
            planning_interval=5,
        )

    def _run_phase(
        self,
        phase_name: str,
        phase_prompt: str,
        tools: list,
        additional_args: dict,
        registry_key: str,
    ) -> bool:
        """
        Run a single evaluation phase with basic validation.

        Args:
            phase_name: Name of the phase
            phase_prompt: Prompt for this phase
            tools: Tools for this phase
            additional_args: Additional args to inject (solution, test_sample_df, etc.)
            registry_key: Key to check in context.scratch for completion

        Returns:
            True if phase completed successfully, False otherwise
        """
        logger.info(f"Starting evaluation phase: {phase_name}")

        agent = self._build_agent(phase_name, phase_prompt, tools)

        try:
            agent.run(
                task=phase_prompt,
                additional_args=additional_args,
            )

            # Check if phase registered output
            if registry_key not in self.context.scratch:
                logger.warning(f"Phase {phase_name} completed but {registry_key} not found in scratch")
                return False

            logger.info(f"Phase {phase_name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Phase {phase_name} failed: {e}")
            return False

    @agent_span("ModelEvaluatorAgent")
    def run(
        self,
        solution: Solution,
        test_sample_df: pd.DataFrame,
        predictor: Any,
    ) -> EvaluationReport | None:
        """
        Execute multi-phase evaluation.

        Args:
            solution: Solution object to evaluate
            test_sample_df: Test dataset sample (pandas DataFrame)
            predictor: Trained predictor instance

        Returns:
            EvaluationReport or None if evaluation fails
        """
        logger.info("Starting multi-phase model evaluation")

        # Prepare additional_args for object injection
        additional_args = {
            "solution": solution,
            "test_sample_df": test_sample_df,
            "predictor": predictor,
            "primary_metric_name": self.context.metric.name,
            "output_targets": self.context.output_targets,  # Target column names (for all phases)
        }

        explainability_required = solution.model_type in ["xgboost", "catboost"]

        # Phase 1: Core Metrics
        success = self._run_phase(
            phase_name="CoreMetrics",
            phase_prompt=self._get_phase_1_prompt(self.context.intent, self.context.metric.name),
            tools=[get_register_core_metrics_tool(self.context)],
            additional_args=additional_args,
            registry_key="_core_metrics_report",
        )
        if not success:
            logger.error("Core Metrics phase failed - cannot continue evaluation")
            return None

        # Phase 2: Error Analysis
        success = self._run_phase(
            phase_name="ErrorAnalysis",
            phase_prompt=self._get_phase_2_prompt(self.context.intent),
            tools=[get_register_diagnostic_report_tool(self.context)],
            additional_args=additional_args,
            registry_key="_diagnostic_report",
        )
        if not success:
            logger.warning("Error Analysis phase failed - continuing with partial evaluation")

        # Phase 3: Robustness
        success = self._run_phase(
            phase_name="Robustness",
            phase_prompt=self._get_phase_3_prompt(self.context.intent),
            tools=[get_register_robustness_report_tool(self.context)],
            additional_args=additional_args,
            registry_key="_robustness_report",
        )
        if not success:
            logger.warning("Robustness phase failed - continuing with partial evaluation")

        # Phase 4: Explainability (conditional)
        if explainability_required:
            success = self._run_phase(
                phase_name="Explainability",
                phase_prompt=self._get_phase_4_prompt(self.context.intent),
                tools=[get_register_explainability_report_tool(self.context)],
                additional_args=additional_args,
                registry_key="_explainability_report",
            )
            if not success:
                logger.warning("Explainability phase failed - continuing with partial evaluation")

        # Phase 5: Baseline Comparison
        # Inject baseline info + baseline predictor for re-evaluation on test set
        baseline_context = {
            "baseline_name": self.context.heuristic_baseline.name if self.context.heuristic_baseline else "unknown",
            "baseline_type": "heuristic",
            "baseline_description": (
                self.context.heuristic_baseline.metadata.get("description", "")
                if self.context.heuristic_baseline
                else ""
            ),
            "baseline_performance": self.context.baseline_performance,  # Validation set performance (for reference)
            "baseline_predictor": self.context.baseline_predictor,  # For re-evaluation on test set
        }
        synthesis_args = {**additional_args, **baseline_context}

        success = self._run_phase(
            phase_name="BaselineComparison",
            phase_prompt=self._get_phase_baseline_comparison_prompt(self.context.intent),
            tools=[get_register_baseline_comparison_tool(self.context)],
            additional_args=synthesis_args,
            registry_key="_baseline_comparison_report",
        )
        if not success:
            logger.warning("Baseline Comparison phase failed - continuing with partial evaluation")

        # Phase 6: Synthesis
        # Inject component reports
        core_metrics = self.context.scratch.get("_core_metrics_report")
        diagnostics = self.context.scratch.get("_diagnostic_report")
        robustness = self.context.scratch.get("_robustness_report")
        explainability = self.context.scratch.get("_explainability_report") if explainability_required else None
        baseline_comparison = self.context.scratch.get("_baseline_comparison_report")

        synthesis_args = {
            **additional_args,
            "core_metrics_report": core_metrics,
            "diagnostic_report": diagnostics,
            "robustness_report": robustness,
            "explainability_report": explainability,
            "baseline_comparison_report": baseline_comparison,
        }

        success = self._run_phase(
            phase_name="Synthesis",
            phase_prompt=self._get_phase_synthesis_prompt(self.context.intent, explainability_required),
            tools=[get_register_final_evaluation_tool(self.context)],
            additional_args=synthesis_args,
            registry_key="_evaluation_report",
        )

        if not success:
            logger.error("Synthesis phase failed - cannot create final evaluation report")
            # Cannot create valid EvaluationReport without all required components
            # Return None to signal failure
            return None

        # Get final report
        evaluation_report = self.context.scratch.get("_evaluation_report")
        logger.info("Multi-phase model evaluation completed")
        return evaluation_report

    # ============================================
    # Phase Prompts
    # ============================================

    @staticmethod
    def _get_phase_1_prompt(task: str, primary_metric_name: str) -> str:
        return (
            f"PHASE 1: CORE METRICS EVALUATION\\n\\n"
            f"Task Context: {task}\\n"
            f"Primary Metric: {primary_metric_name}\\n\\n"
            f"Your mission: Compute comprehensive performance metrics on the test sample.\\n\\n"
            f"Write code to:\\n"
            f"1. Detect task type by examining test_sample_df and target variable\\n"
            f"2. Select appropriate metrics for this task type\\n"
            f"3. Compute primary metric + 4-6 additional relevant metrics\\n"
            f"4. ENCOURAGED: Compute 95% confidence intervals using bootstrap (1000 samples)\\n"
            f"5. Interpret results - what do these numbers tell us?\\n\\n"
            f"Register using:\\n"
            f"register_core_metrics_report(\\n"
            f"    task_type='...',  # your detected type\\n"
            f"    primary_metric_name='{primary_metric_name}',\\n"
            f"    primary_metric_value=...,\\n"
            f"    primary_metric_ci_lower=...,  # None if not computed\\n"
            f"    primary_metric_ci_upper=...,  # None if not computed\\n"
            f"    all_metrics={{...}},\\n"
            f"    statistical_notes='Your interpretation',\\n"
            f"    metric_confidence_intervals=None,  # Optional\\n"
            f"    visualizations=None  # Optional (headless)\\n"
            f")\\n\\n"
            f"After successful registration, call final_answer('Phase 1 complete').\\n\\n"
            f"IMPORTANT: Focus on rigorous computation and thoughtful interpretation."
        )

    @staticmethod
    def _get_phase_2_prompt(task: str) -> str:
        return (
            f"PHASE 2: ERROR ANALYSIS\\n\\n"
            f"Task Context: {task}\\n\\n"
            f"Your mission: Identify and explain failure patterns.\\n\\n"
            f"Write code to:\\n"
            f"1. Identify top 20-30 worst predictions by ABSOLUTE error with feature context\\n"
            f"2. Detect error patterns (feature correlations, clustering, etc.)\\n"
            f"3. Analyze subpopulations if applicable\\n"
            f"4. Synthesize insights - WHY is the model failing?\\n\\n"
            f"Register using:\\n"
            f"register_diagnostic_report(\\n"
            f"    worst_predictions=[{{index, true_value, predicted_value, error, features}}, ...],\\n"
            f"    error_patterns=['Pattern 1', ...],\\n"
            f"    key_insights=['Insight 1', ...],\\n"
            f"    error_distribution_summary='Your summary',\\n"
            f"    subgroup_analysis=None  # Optional\\n"
            f")\\n\\n"
            f"After successful registration, call final_answer('Phase 2 complete').\\n\\n"
            f"IMPORTANT: Focus on pattern detection and explaining WHY failures occur."
        )

    @staticmethod
    def _get_phase_3_prompt(task: str) -> str:
        return (
            f"PHASE 3: ROBUSTNESS ASSESSMENT\\n\\n"
            f"Task Context: {task}\\n\\n"
            f"Your mission: Stress-test model reliability.\\n\\n"
            f"Write code to test:\\n"
            f"1. Perturbation sensitivity (noise, missing values, boundary values)\\n"
            f"2. Consistency (same input → same output?)\\n"
            f"3. Grade robustness A-F based on results\\n"
            f"4. Identify specific concerns\\n\\n"
            f"Register using:\\n"
            f"register_robustness_report(\\n"
            f"    perturbation_tests={{...}},\\n"
            f"    robustness_grade='A'|'B'|'C'|'D'|'F',\\n"
            f"    concerns=[...],\\n"
            f"    recommendations=[...],\\n"
            f"    consistency_score=None  # Optional 0-1 score\\n"
            f")\\n\\n"
            f"After successful registration, call final_answer('Phase 3 complete').\\n\\n"
            f"IMPORTANT: Focus on uncovering risks and stress-testing reliability."
        )

    @staticmethod
    def _get_phase_4_prompt(task: str) -> str:
        return (
            f"PHASE 4: EXPLAINABILITY ANALYSIS\\n\\n"
            f"Task Context: {task}\\n\\n"
            f"Your mission: Explain which features drive predictions.\\n\\n"
            f"Write code to:\\n"
            f"1. Select appropriate method (built-in, permutation, SHAP)\\n"
            f"2. Compute feature importance for all features\\n"
            f"3. Interpret results - what do these features mean?\\n\\n"
            f"Register using:\\n"
            f"register_explainability_report(\\n"
            f"    feature_importance={{...}},\\n"
            f"    method_used='...',\\n"
            f"    top_features=[...],\\n"
            f"    interpretation='Your explanation',\\n"
            f"    confidence_intervals=None  # Optional\\n"
            f")\\n\\n"
            f"After successful registration, call final_answer('Phase 4 complete').\\n\\n"
            f"Focus on meaningful interpretation of feature importance."
        )

    @staticmethod
    def _get_phase_baseline_comparison_prompt(task: str) -> str:
        return (
            f"PHASE 5: BASELINE COMPARISON\\n\\n"
            f"Task Context: {task}\\n\\n"
            f"Your mission: Compare the trained model against the baseline predictor.\\n\\n"
            f"The following baseline info is available as variables:\\n"
            f"- baseline_name: str\\n"
            f"- baseline_type: str\\n"
            f"- baseline_description: str\\n"
            f"- baseline_performance: float (validation set performance - for reference)\\n"
            f"- baseline_predictor: HeuristicBaselinePredictor instance with .predict(X) method\\n"
            f"- output_targets: list[str] (target column names)\\n"
            f"- core_metrics_report: CoreMetricsReport from Phase 1 (model's test metrics)\\n\\n"
            f"BASELINE PREDICTOR INTERFACE:\\n"
            f"The baseline_predictor.predict() takes DataFrame (features only) and returns numpy array.\\n"
            f"Example: y_pred_baseline = baseline_predictor.predict(X_test)\\n\\n"
            f"Write code to:\\n"
            f"1. Extract model test metrics from core_metrics_report.all_metrics\\n"
            f"2. Evaluate baseline on test_sample_df to get baseline test metrics\\n"
            f"   (prepare X_test by dropping output_targets columns, call baseline_predictor.predict())\\n"
            f"3. Calculate absolute deltas (model - baseline) for each metric\\n"
            f"4. Calculate percentage changes where applicable\\n"
            f"5. Interpret: Does model improvement justify its complexity?\\n\\n"
            f"Register using:\\n"
            f"register_baseline_comparison_report(\\n"
            f"    baseline_name=baseline_name,\\n"
            f"    baseline_type=baseline_type,\\n"
            f"    baseline_description=baseline_description,\\n"
            f"    baseline_performance={{metric: value}},  # Baseline on test set\\n"
            f"    model_performance={{metric: value}},  # Model on test set\\n"
            f"    performance_delta={{metric: model - baseline}},\\n"
            f"    interpretation='Your analysis',\\n"
            f"    performance_delta_pct={{metric: percentage}}  # Optional\\n"
            f")\\n\\n"
            f"After successful registration, call final_answer('Baseline comparison complete').\\n\\n"
            f"Focus on contextualizing model performance relative to the simple baseline."
        )

    @staticmethod
    def _get_phase_synthesis_prompt(task: str, explainability_required: bool) -> str:
        explainability_note = "- explainability_report: ExplainabilityReport" if explainability_required else ""

        return (
            f"PHASE 6: SYNTHESIS\\n\\n"
            f"Task Context: {task}\\n\\n"
            f"Your mission: Compose final evaluation report and verdict.\\n\\n"
            f"Component reports are already available as variables:\\n"
            f"- core_metrics_report: CoreMetricsReport\\n"
            f"- diagnostic_report: DiagnosticReport\\n"
            f"- robustness_report: RobustnessReport\\n"
            f"- baseline_comparison_report: BaselineComparisonReport\\n"
            f"{explainability_note}\\n\\n"
            f"You can access them directly (e.g., core_metrics_report.primary_metric_value if not None).\\n\\n"
            f"Write code to:\\n"
            f"1. Analyze component reports to determine verdict (PASS/CONDITIONAL_PASS/FAIL)\\n"
            f"2. Synthesize prioritized recommendations (HIGH/MEDIUM/LOW)\\n"
            f"3. Write executive summary (2-3 sentences)\\n"
            f"4. Determine deployment readiness\\n\\n"
            f"Register using:\\n"
            f"register_final_evaluation_report(\\n"
            f"    verdict='PASS'|'CONDITIONAL_PASS'|'FAIL',\\n"
            f"    summary='...',\\n"
            f"    deployment_ready=True|False,\\n"
            f"    key_concerns=[...],\\n"
            f"    recommendations=[{{'priority': 'HIGH', 'action': '...', 'rationale': '...'}}]\\n"
            f")\\n\\n"
            f"After registration, call final_answer('Evaluation complete').\\n\\n"
            f"Focus on clear decision-making and actionable guidance."
        )
