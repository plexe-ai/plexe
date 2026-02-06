"""Overview tab: Phase progress, timeline, key metrics."""

import streamlit as st


def render_overview(exp_meta, checkpoints):
    """Render overview tab."""

    # Extract performance data
    context_data = {}
    for checkpoint in checkpoints.values():
        ctx = checkpoint.get("context") or {}
        context_data.update(ctx)

    baseline_data = context_data.get("heuristic_baseline") or {}
    baseline_perf = baseline_data.get("performance")
    best_perf = exp_meta.best_performance

    # Key metrics (top row)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", exp_meta.status.title())

    with col2:
        st.metric("Current Phase", exp_meta.current_phase or "Not started")

    with col3:
        if exp_meta.metric_name:
            st.metric("Metric", exp_meta.metric_name)
        else:
            st.metric("Metric", "N/A")

    with col4:
        # Show baseline → best with delta
        if best_perf is not None and baseline_perf is not None:
            delta = best_perf - baseline_perf
            st.metric("Best Performance", f"{best_perf:.4f}", delta=f"{delta:+.4f}")
        elif best_perf is not None:
            st.metric("Best Performance", f"{best_perf:.4f}")
        else:
            st.metric("Best Performance", "N/A")

    # Phase timeline (compact table)
    st.subheader("Phase Timeline")

    phases = [
        ("analyze_data", "Data Understanding"),
        ("prepare_data", "Data Preparation"),
        ("build_baselines", "Baselines"),
        ("search_models", "Model Search"),
        ("evaluate_final", "Evaluation"),
        ("package_final_model", "Package"),
    ]

    timeline_data = []
    for i, (phase_key, phase_name) in enumerate(phases, 1):
        checkpoint = checkpoints.get(phase_key)

        if checkpoint:
            status = checkpoint.get("status", "completed")
            timestamp = checkpoint.get("timestamp", "")[:19]  # Trim to datetime
            timeline_data.append(
                {
                    "Phase": f"{i}. {phase_name}",
                    "Status": status.title(),
                    "Completed": timestamp if status == "completed" else "-",
                }
            )
        else:
            timeline_data.append({"Phase": f"{i}. {phase_name}", "Status": "Pending", "Completed": "-"})

    st.dataframe(timeline_data, use_container_width=True, hide_index=True)

    # Experiment info (compact)
    st.subheader("Details")

    col1, col2 = st.columns(2)

    with col1:
        st.text(f"Experiment ID: {exp_meta.experiment_id}")
        st.text(f"Dataset: {exp_meta.dataset_name}")
        st.text(f"Timestamp: {exp_meta.timestamp}")

    with col2:
        st.text(f"Path: {exp_meta.path.name}")
        st.text(f"Last Modified: {exp_meta.last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
