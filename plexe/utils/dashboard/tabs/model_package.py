"""Model Package tab: File structure, metadata."""

import streamlit as st
import yaml

from plexe.utils.dashboard.utils import load_code_file, load_json_file


def render_model_package(exp_path):
    """Render model package tab."""

    model_dir = exp_path / "model"
    if not model_dir.exists():
        st.warning("Model not yet packaged")
        return

    model_yaml = model_dir / "model.yaml"
    if model_yaml.exists():
        with open(model_yaml) as f:
            metadata = yaml.safe_load(f)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model Type", metadata.get("model_type", "Unknown"))

        with col2:
            st.metric("Task Type", metadata.get("task_type", "Unknown"))

        with col3:
            metric_info = metadata.get("metric", {})
            if isinstance(metric_info, dict):
                metric_name = metric_info.get("name", "Unknown")
                st.metric("Metric", metric_name)
            else:
                st.metric("Metric", "N/A")

        with col4:
            metric_info = metadata.get("metric", {})
            if isinstance(metric_info, dict):
                metric_value = metric_info.get("value")
                st.metric("Performance", f"{metric_value:.4f}" if metric_value is not None else "N/A")
            else:
                st.metric("Performance", "N/A")

    # Schemas
    st.subheader("Input/Output Schemas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Schema**")
        input_schema = load_json_file(model_dir / "schemas" / "input.json")
        if input_schema:
            properties = input_schema.get("properties", {})
            schema_data = [{"Field": k, "Type": v.get("type", "unknown")} for k, v in properties.items()]
            st.dataframe(schema_data[:20], use_container_width=True, hide_index=True)
            if len(schema_data) > 20:
                st.caption(f"Showing 20 of {len(schema_data)} fields")

    with col2:
        st.markdown("**Output Schema**")
        output_schema = load_json_file(model_dir / "schemas" / "output.json")
        if output_schema:
            properties = output_schema.get("properties", {})
            schema_data = [{"Field": k, "Type": v.get("type", "unknown")} for k, v in properties.items()]
            st.dataframe(schema_data, use_container_width=True, hide_index=True)

    # Predictor code
    st.subheader("Predictor Code")

    predictor_code = load_code_file(model_dir / "predictor.py")
    if predictor_code:
        st.code(predictor_code, language="python", line_numbers=True)

    # Final pipeline code
    st.subheader("Feature Engineering Code")

    pipeline_code = load_code_file(model_dir / "src" / "pipeline.py")
    if pipeline_code:
        st.code(pipeline_code, language="python", line_numbers=True)

    # Package info
    st.subheader("Package")

    tarball = exp_path / "model.tar.gz"
    if tarball.exists():
        size_mb = tarball.stat().st_size / (1024**2)
        st.text(f"Archive: model.tar.gz ({size_mb:.2f} MB)")

    with st.expander("File Tree"):
        _render_file_tree(model_dir)


def _render_file_tree(directory, prefix=""):
    """Recursively render file tree."""

    items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))

    for item in items:
        if item.is_dir():
            st.write(f"{prefix}**{item.name}/**")
            _render_file_tree(item, prefix + "  ")
        else:
            size_kb = item.stat().st_size / 1024
            st.write(f"{prefix}{item.name} ({size_kb:.1f} KB)")
