"""Evaluation tab: Final test metrics, baseline comparison, diagnostics."""

import plotly.graph_objects as go
import streamlit as st

from plexe.utils.dashboard.utils import load_report


def render_evaluation(checkpoints, exp_path):
    """Render evaluation tab."""

    checkpoint = checkpoints.get("evaluate_final")
    if not checkpoint:
        st.warning("Phase 5 not yet completed")
        return

    context = checkpoint.get("context") or {}

    # Load evaluation report from DirNames.BUILD_DIR/reports/
    eval_report_file = load_report(exp_path, "05_final_evaluation")

    if eval_report_file:
        _render_full_evaluation(eval_report_file)
    else:
        # Fallback to context
        scratch = context.get("scratch") or {}
        eval_report = scratch.get("_evaluation_report") or {}

        if eval_report:
            _render_full_evaluation(eval_report)
        else:
            _render_basic_metrics(context)


def _render_full_evaluation(eval_report):
    """Render full evaluation report with visualizations."""

    # Verdict
    col1, col2 = st.columns(2)

    with col1:
        verdict = eval_report.get("verdict", "Unknown")
        st.metric("Verdict", verdict)

    with col2:
        deployment_ready = eval_report.get("deployment_ready", False)
        st.metric("Deployment Ready", "Yes" if deployment_ready else "No")

    summary = eval_report.get("summary", "")
    if summary:
        st.info(summary)

    # All metrics grid
    core_metrics = eval_report.get("core_metrics") or {}
    all_metrics = core_metrics.get("all_metrics") or {}

    if all_metrics:
        st.subheader("Performance Metrics")
        cols = st.columns(min(len(all_metrics), 6))
        for idx, (metric_name, metric_value) in enumerate(all_metrics.items()):
            with cols[idx % 6]:
                st.metric(metric_name, f"{metric_value:.4f}")

    # Feature importance chart
    explainability = eval_report.get("explainability") or {}
    feature_importance = explainability.get("feature_importance") or {}

    if feature_importance:
        st.subheader("Feature Importance")

        # Sort and take top 20
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        features = [f for f, _ in sorted_features]
        importances = [imp for _, imp in sorted_features]

        fig = go.Figure(data=[go.Bar(y=features, x=importances, orientation="h", marker=dict(color="#2563eb"))])

        fig.update_layout(
            height=max(300, len(features) * 20),
            template="plotly_white",
            margin=dict(l=150, r=20, t=20, b=40),
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Baseline comparison
    baseline_comparison = eval_report.get("baseline_comparison") or {}
    if baseline_comparison:
        st.subheader("Baseline Comparison")

        baseline_perf = baseline_comparison.get("baseline_performance") or {}
        model_perf = baseline_comparison.get("model_performance") or {}
        delta = baseline_comparison.get("performance_delta") or {}

        if baseline_perf and model_perf:
            comparison_data = []
            for metric in baseline_perf.keys():
                comparison_data.append(
                    {
                        "Metric": metric,
                        "Baseline": f"{baseline_perf[metric]:.4f}",
                        "Model": f"{model_perf[metric]:.4f}",
                        "Delta": (
                            f"+{delta.get(metric, 0):.4f}"
                            if delta.get(metric, 0) >= 0
                            else f"{delta.get(metric, 0):.4f}"
                        ),
                    }
                )

            st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    # Worst predictions
    diagnostics = eval_report.get("diagnostics") or {}
    worst_predictions = diagnostics.get("worst_predictions", [])

    if worst_predictions:
        st.subheader("Worst Predictions")

        # Show top 10
        worst_df_data = []
        for pred in worst_predictions[:10]:
            worst_df_data.append(
                {
                    "Index": pred.get("index"),
                    "True": pred.get("true_value"),
                    "Predicted": pred.get("predicted_value"),
                    "Error": f"{pred.get('error', 0):.4f}",
                }
            )

        st.dataframe(worst_df_data, use_container_width=True, hide_index=True)

    # Error patterns
    error_patterns = diagnostics.get("error_patterns", [])
    if error_patterns:
        st.subheader("Error Patterns")
        for pattern in error_patterns:
            st.text(f"• {pattern}")

    # Robustness
    robustness = eval_report.get("robustness") or {}
    if robustness:
        st.subheader("Robustness Assessment")

        col1, col2 = st.columns(2)
        with col1:
            grade = robustness.get("robustness_grade", "N/A")
            st.metric("Grade", grade)

        with col2:
            concerns = robustness.get("concerns", [])
            st.metric("Concerns", len(concerns))

        if concerns:
            with st.expander("Concerns"):
                for concern in concerns:
                    st.text(f"• {concern}")

    # Recommendations
    recommendations = eval_report.get("recommendations", [])
    if recommendations:
        st.subheader("Recommendations")

        rec_data = []
        for rec in recommendations:
            rec_data.append(
                {
                    "Priority": rec.get("priority", "MEDIUM"),
                    "Action": rec.get("action", ""),
                }
            )

        st.dataframe(rec_data, use_container_width=True, hide_index=True)


def _render_basic_metrics(context):
    """Render basic metrics if full report not available."""

    st.info("Full evaluation report not available")

    baseline = context.get("heuristic_baseline") or {}
    baseline_perf = baseline.get("performance")

    if baseline_perf is not None:
        st.metric("Baseline Performance", f"{baseline_perf:.4f}")
