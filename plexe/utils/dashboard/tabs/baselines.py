"""Baselines tab: Heuristic baseline info."""

import streamlit as st

from plexe.constants import DirNames
from plexe.utils.dashboard.utils import load_code_file


def render_baselines(checkpoints, exp_path):
    """Render baselines tab."""

    checkpoint = checkpoints.get("build_baselines")
    if not checkpoint:
        st.warning("Phase 3 not yet completed")
        return

    context = checkpoint.get("context") or {}
    baseline_data = context.get("heuristic_baseline") or {}

    if not baseline_data:
        st.info("No baseline data available")
        return

    name = baseline_data.get("name", "Unknown")
    model_type = baseline_data.get("model_type", "Unknown")
    performance = baseline_data.get("performance")

    # Performance display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Name", name)

    with col2:
        st.metric("Type", model_type)

    with col3:
        st.metric("Performance", f"{performance:.4f}" if performance is not None else "N/A")

    # Baseline code
    st.subheader("Implementation")

    baseline_code_path = exp_path / DirNames.BUILD_DIR / "search" / "baselines" / f"{name}.py"
    baseline_code = load_code_file(baseline_code_path)

    if baseline_code:
        st.code(baseline_code, language="python", line_numbers=True)
    else:
        st.info(f"Code not found at {baseline_code_path}")

    # Metadata
    metadata = baseline_data.get("metadata") or {}
    if metadata:
        with st.expander("Metadata"):
            for key, value in metadata.items():
                st.text(f"{key}: {value}")
