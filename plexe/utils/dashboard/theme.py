"""Custom theme and styling for dashboard."""

import streamlit as st


def apply_custom_theme():
    """Apply custom CSS for dense, professional layout."""

    st.markdown(
        """
        <style>
        /* Dense layout */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }

        /* Compact metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }

        /* Streamlined headers */
        h1 {
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
            padding-bottom: 0.5rem;
        }

        h2 {
            font-size: 1.1rem;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        h3 {
            font-size: 0.95rem;
            margin-top: 0.75rem;
            margin-bottom: 0.35rem;
        }

        /* Compact tabs */
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
        }

        /* Tight buttons */
        .stButton button {
            padding: 0.3rem 0.6rem;
            font-size: 0.8rem;
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton button {
            font-size: 0.75rem;
            padding: 0.35rem 0.5rem;
            font-family: monospace;
        }

        /* Compact expanders */
        .streamlit-expanderHeader {
            font-size: 0.8rem;
            padding: 0.4rem 0.6rem;
        }

        /* Dense code blocks */
        pre {
            font-size: 0.7rem;
            padding: 0.4rem;
            line-height: 1.3;
        }

        /* Minimal spacing */
        .element-container {
            margin-bottom: 0.35rem;
        }

        /* Small captions */
        [data-testid="stCaptionContainer"] {
            font-size: 0.7rem;
        }

        /* Compact dividers */
        hr {
            margin: 0.75rem 0;
        }

        /* Dense dataframes */
        [data-testid="stDataFrame"] {
            font-size: 0.8rem;
        }

        /* Tight alerts */
        [data-testid="stAlert"] {
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
