"""Main Streamlit dashboard application."""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st

from plexe.utils.dashboard.discovery import discover_experiments, load_experiment_checkpoints
from plexe.utils.dashboard.theme import apply_custom_theme

# Page config
st.set_page_config(
    page_title="Model Builder Dashboard",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
apply_custom_theme()

# Load config from environment
WORK_DIR = Path(os.environ.get("MBS_WORK_DIR", "./workdir"))
REFRESH_INTERVAL = int(os.environ.get("MBS_REFRESH_INTERVAL", "2"))

# Initialize session state
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ============================================
# Sidebar: Experiment Selector
# ============================================

st.sidebar.title("Experiments")

# Discover experiments
try:
    experiments = discover_experiments(WORK_DIR)
except Exception as e:
    st.sidebar.error(f"Error discovering experiments: {e}")
    st.stop()

if not experiments:
    st.sidebar.warning(f"No experiments found in {WORK_DIR}")
    st.sidebar.info("Run model builder to create experiments")
    st.stop()

# Group experiments by dataset
datasets = defaultdict(list)
for exp in experiments:
    datasets[exp.dataset_name].append(exp)

# Render grouped experiments
for dataset_name, dataset_exps in datasets.items():
    st.sidebar.markdown(f"**{dataset_name}**")

    for exp in dataset_exps:
        is_selected = st.session_state.get("selected_exp_path") == str(exp.path)

        # Show timestamp with status indicator
        status_indicators = {"completed": "✓", "running": "●", "failed": "✗", "paused": "○"}
        indicator = status_indicators.get(exp.status, "?")

        button_label = f"  {indicator} {exp.timestamp[:10]}..."

        if st.sidebar.button(
            button_label,
            key=f"exp_{exp.path}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state.selected_exp_path = str(exp.path)
            st.session_state.selected_exp_meta = exp
            st.rerun()

        # Show quick stats if selected
        if is_selected:
            perf_str = f"{exp.best_performance:.4f}" if exp.best_performance is not None else "N/A"
            st.sidebar.caption(f"   Phase {exp.phase_number}/6 · {perf_str}")

    st.sidebar.divider()

# Refresh indicator and auto-refresh
st.sidebar.divider()
time_since = (datetime.now() - st.session_state.last_refresh).total_seconds()
st.sidebar.caption(f"Refreshed {int(time_since)}s ago")

# Auto-refresh toggle
if st.sidebar.checkbox("Auto-refresh", value=True):
    if time_since >= REFRESH_INTERVAL:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# ============================================
# Main Area: Experiment Details
# ============================================

# Check if experiment is selected
if "selected_exp_path" not in st.session_state:
    st.info("Select an experiment from the sidebar")
    st.stop()

try:
    # Load selected experiment
    exp_meta = st.session_state.selected_exp_meta
    exp_path = Path(st.session_state.selected_exp_path)
    checkpoints = load_experiment_checkpoints(exp_path)

    # Header (compact)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{exp_meta.dataset_name} / {exp_meta.timestamp}")
        if exp_meta.intent:
            st.caption(exp_meta.intent)
    with col2:
        status_color = {"completed": "green", "running": "blue", "failed": "red", "paused": "gray"}.get(
            exp_meta.status, "gray"
        )
        st.markdown(f"**Status:** :{status_color}[{exp_meta.status.upper()}]")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Data Understanding",
            "Data Preparation",
            "Baselines",
            "Search Tree",
            "Evaluation",
            "Model Package",
        ]
    )

    # Tab content (import from separate modules)
    with tab1:
        from plexe.utils.dashboard.tabs.overview import render_overview

        render_overview(exp_meta, checkpoints)

    with tab2:
        from plexe.utils.dashboard.tabs.data_understanding import render_data_understanding

        render_data_understanding(checkpoints, exp_path)

    with tab3:
        from plexe.utils.dashboard.tabs.data_preparation import render_data_preparation

        render_data_preparation(checkpoints, exp_path)

    with tab4:
        from plexe.utils.dashboard.tabs.baselines import render_baselines

        render_baselines(checkpoints, exp_path)

    with tab5:
        from plexe.utils.dashboard.tabs.search_tree import render_search_tree

        render_search_tree(checkpoints, exp_path)

    with tab6:
        from plexe.utils.dashboard.tabs.evaluation import render_evaluation

        render_evaluation(checkpoints, exp_path)

    with tab7:
        from plexe.utils.dashboard.tabs.model_package import render_model_package

        render_model_package(exp_path)

except Exception as e:
    st.error(f"Error rendering dashboard: {e}")
    st.exception(e)
