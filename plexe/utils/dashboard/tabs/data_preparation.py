"""Data Preparation tab: Splits, sample sizes, data preview."""

import streamlit as st

from plexe.utils.dashboard.utils import load_parquet_sample, get_parquet_row_count


def render_data_preparation(checkpoints, exp_path):
    """Render data preparation tab."""

    checkpoint = checkpoints.get("prepare_data")
    if not checkpoint:
        st.warning("Phase 2 not yet completed")
        return

    context = checkpoint.get("context") or {}

    train_uri = context.get("train_uri")
    val_uri = context.get("val_uri")
    test_uri = context.get("test_uri")
    train_sample_uri = context.get("train_sample_uri")
    val_sample_uri = context.get("val_sample_uri")

    # Split sizes table with row counts
    st.subheader("Dataset Splits")

    split_data = []
    for split_name, full_uri, sample_uri in [
        ("Train", train_uri, train_sample_uri),
        ("Validation", val_uri, val_sample_uri),
        ("Test", test_uri, None),
    ]:
        full_count = get_parquet_row_count(full_uri) if full_uri else None
        sample_count = get_parquet_row_count(sample_uri) if sample_uri else None

        split_data.append(
            {
                "Split": split_name,
                "Full Dataset": f"{full_count:,}" if full_count else "N/A",
                "Sample": f"{sample_count:,}" if sample_count else "-",
            }
        )

    st.dataframe(split_data, use_container_width=True, hide_index=True)

    # Data preview
    st.subheader("Data Preview (Train Sample)")

    if train_sample_uri:
        sample_df = load_parquet_sample(train_sample_uri, limit=10)
        if sample_df is not None:
            st.dataframe(sample_df, use_container_width=True)
            st.caption(f"Showing 10 of {get_parquet_row_count(train_sample_uri) or '?'} rows")
        else:
            st.info("Could not load data preview")

    # URIs in expander
    with st.expander("Dataset URIs"):
        st.text(f"Train: {train_uri}")
        st.text(f"Val: {val_uri}")
        st.text(f"Test: {test_uri or 'N/A'}")
        st.text(f"Train Sample: {train_sample_uri}")
        st.text(f"Val Sample: {val_sample_uri}")
