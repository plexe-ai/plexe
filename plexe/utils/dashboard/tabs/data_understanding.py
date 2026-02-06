"""Data Understanding tab: Layout, stats, task analysis, metric."""

import plotly.graph_objects as go
import streamlit as st

from plexe.utils.dashboard.utils import load_report


def render_data_understanding(checkpoints, exp_path):
    """Render data understanding tab."""

    checkpoint = checkpoints.get("analyze_data")
    if not checkpoint:
        st.warning("Phase 1 not yet completed")
        return

    context = checkpoint.get("context") or {}

    # Load reports for richer data
    stats_report = load_report(exp_path, "01_statistical_analysis")
    task_report = load_report(exp_path, "02_task_analysis")

    # Top summary row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        layout = context.get("data_layout", "Unknown")
        st.metric("Layout", layout)

    with col2:
        rows = stats_report.get("total_rows") if stats_report else context.get("stats", {}).get("num_rows")
        st.metric("Rows", f"{rows:,}" if rows else "N/A")

    with col3:
        cols = stats_report.get("total_columns") if stats_report else context.get("stats", {}).get("num_columns")
        st.metric("Columns", cols or "N/A")

    with col4:
        targets = context.get("output_targets", [])
        st.metric("Target", targets[0] if targets else "N/A")

    with col5:
        task_type = task_report.get("task_type") if task_report else context.get("task_analysis", {}).get("task_type")
        st.metric("Task", task_type or "N/A")

    # Column breakdown and missing values side by side
    if stats_report:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Column Types")
            type_counts = {
                "Numeric": len(stats_report.get("numeric_columns", [])),
                "String": len(stats_report.get("string_columns", [])),
                "Other": len(stats_report.get("other_columns", [])),
            }
            fig = go.Figure(data=[go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()))])
            fig.update_layout(
                height=250, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Missing Values")
            missing_summary = stats_report.get("missing_value_summary", {})
            if missing_summary:
                # Sort by missing % and show top 10
                sorted_missing = sorted(missing_summary.items(), key=lambda x: x[1], reverse=True)[:10]
                cols_list = [col for col, _ in sorted_missing]
                pcts = [pct * 100 for _, pct in sorted_missing]

                fig = go.Figure(data=[go.Bar(x=cols_list, y=pcts, orientation="v")])
                fig.update_layout(
                    height=250,
                    yaxis_title="Missing %",
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=100),
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    # Target distribution (for classification)
    if task_report and task_report.get("task_type") in ["binary_classification", "multiclass_classification"]:
        st.subheader("Target Distribution")

        target_desc = task_report.get("target_description", {})
        distribution = target_desc.get("distribution", {})

        if distribution:
            # Extract class counts
            class_data = []
            for key, value in distribution.items():
                if key.startswith("class_") and not key.endswith("_pct"):
                    class_label = key
                    count = value
                    pct = distribution.get(f"{key}_pct", 0)
                    class_data.append({"Class": class_label, "Count": count, "Percentage": f"{pct:.1f}%"})

            if class_data:
                st.dataframe(class_data, use_container_width=True, hide_index=True)

                # Balance indicator
                balance = target_desc.get("balance", "Unknown")
                st.caption(f"Balance: {balance}")

    # Metric selection
    st.subheader("Metric Selection")

    metric_data = context.get("metric") or {}
    if metric_data:
        metric_name = metric_data.get("name", "Unknown")
        optimization_direction = metric_data.get("optimization_direction", "Unknown")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Metric", metric_name)
        with col2:
            st.metric("Direction", optimization_direction)

        # Rationale
        scratch = context.get("scratch") or {}
        metric_selection = scratch.get("_metric_selection") or {}
        rationale = metric_selection.get("rationale")
        if rationale:
            with st.expander("Rationale"):
                st.write(rationale)

    # Full reports in expanders
    with st.expander("Statistical Analysis Report"):
        if stats_report:
            st.json(stats_report)

    with st.expander("Task Analysis Report"):
        if task_report:
            st.json(task_report)
