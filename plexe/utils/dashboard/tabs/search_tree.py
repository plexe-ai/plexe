"""Search Tree tab: Tree visualization, performance chart, insights."""

import plotly.graph_objects as go
import streamlit as st

from plexe.constants import DirNames
from plexe.utils.dashboard.utils import load_code_file


def render_search_tree(checkpoints, exp_path):
    """Render search tree tab."""

    checkpoint = checkpoints.get("search_models")
    if not checkpoint:
        st.warning("Phase 4 not yet started")
        return

    journal = checkpoint.get("search_journal") or {}
    if not journal:
        st.info("No search data available yet")
        return

    nodes = journal.get("nodes", [])
    if not nodes:
        st.info("No solutions explored yet")
        return

    successful_nodes = [n for n in nodes if n.get("performance") is not None and not n.get("is_buggy", False)]
    failed = len([n for n in nodes if n.get("is_buggy", False)])
    best_perf = journal.get("best_performance")

    # Summary stats (compact)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", len(nodes))

    with col2:
        st.metric("Successful", len(successful_nodes))

    with col3:
        st.metric("Failed", failed)

    with col4:
        st.metric("Best", f"{best_perf:.4f}" if best_perf is not None else "N/A")

    # Performance trend and tree side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Performance Trend")
        if successful_nodes:
            solution_ids = [n["solution_id"] for n in successful_nodes]
            performances = [n["performance"] for n in successful_nodes]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=solution_ids,
                    y=performances,
                    mode="lines+markers",
                    name="Performance",
                    line=dict(color="#2563eb", width=2),
                    marker=dict(size=6),
                )
            )

            fig.update_layout(
                xaxis_title="Solution ID",
                yaxis_title="Performance",
                hovermode="closest",
                height=300,
                template="plotly_white",
                margin=dict(l=40, r=10, t=10, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Search Tree")
        fig, config = _build_tree_graph(nodes)
        st.plotly_chart(fig, use_container_width=True, config=config)

    # Node details (compact selector)
    st.subheader("Solution Details")
    node_options = {
        f"#{n['solution_id']}: {n.get('model_type', 'unknown')} - {n.get('performance', 'N/A')}": n for n in nodes
    }
    selected_label = st.selectbox("Select Solution", list(node_options.keys()), label_visibility="collapsed")

    if selected_label:
        node = node_options[selected_label]
        _render_node_details(node, exp_path)

    # Insights (compact table)
    st.subheader("Insights")

    insight_store = checkpoint.get("insight_store") or {}
    insights = insight_store.get("insights", [])

    if insights:
        insights_data = []
        for insight in insights:
            insights_data.append(
                {
                    "ID": insight.get("id"),
                    "Confidence": insight.get("confidence", "unknown").upper(),
                    "Change": insight.get("change", ""),
                    "Effect": insight.get("effect", ""),
                    "Evidence": ", ".join(map(str, insight.get("supporting_evidence", []))),
                }
            )

        st.dataframe(insights_data, use_container_width=True, hide_index=True)
    else:
        st.info("No insights extracted yet")


def _build_tree_graph(nodes):
    """Build plotly tree graph."""

    # Build parent-child map
    children_map = {}
    for node in nodes:
        parent_id = node.get("parent_solution_id")
        if parent_id is not None:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node["solution_id"])

    # Hierarchical layout with better spacing
    positions = {}

    def layout_subtree(node_id, depth, x_offset, width):
        """Recursively layout tree with centered parent above children."""
        children = children_map.get(node_id, [])

        if not children:
            # Leaf node
            positions[node_id] = (x_offset + width / 2, depth)
            return width

        # Layout children first
        child_width = width / len(children)
        child_x = x_offset

        for child in children:
            layout_subtree(child, depth + 1, child_x, child_width)
            child_x += child_width

        # Center parent above children
        child_positions = [positions[c] for c in children]
        avg_x = sum(p[0] for p in child_positions) / len(child_positions)
        positions[node_id] = (avg_x, depth)

        return width

    # Find roots and layout
    roots = [n["solution_id"] for n in nodes if n.get("parent_solution_id") is None]
    root_width = max(len(nodes) / max(len(roots), 1), 3)
    x_offset = 0

    for root in roots:
        layout_subtree(root, 0, x_offset, root_width)
        x_offset += root_width

    # Build edge shapes with arrows
    edge_shapes = []
    for node in nodes:
        parent_id = node.get("parent_solution_id")
        if parent_id is not None and parent_id in positions:
            x0, y0 = positions[parent_id]
            x1, y1 = positions[node["solution_id"]]

            # Add arrow line
            edge_shapes.append(
                {
                    "type": "line",
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "line": {"color": "#cbd5e1", "width": 2},
                }
            )

    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_hover = []

    for node in nodes:
        x, y = positions[node["solution_id"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node["solution_id"]))

        perf = node.get("performance")
        perf_str = f"{perf:.4f}" if perf is not None else "N/A"
        node_hover.append(
            f"Solution {node['solution_id']}<br>Model: {node.get('model_type', 'unknown')}<br>Performance: {perf_str}"
        )

        # Color by status
        if node.get("is_buggy", False):
            node_colors.append("#dc2626")
        elif perf is not None:
            node_colors.append("#2563eb")
        else:
            node_colors.append("#9ca3af")

    # Create figure
    fig = go.Figure()

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=50, color=node_colors, line=dict(color="white", width=3)),
            text=node_text,
            textfont=dict(color="white", size=12, family="monospace"),
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        shapes=edge_shapes,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        height=600,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        dragmode="pan",
    )

    # Enable zoom and pan
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    }

    return fig, config


def _render_node_details(node, exp_path):
    """Render details for a single node."""

    solution_id = node.get("solution_id")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Solution ID", solution_id)

    with col2:
        st.metric("Model", node.get("model_type", "Unknown"))

    with col3:
        perf = node.get("performance")
        st.metric("Performance", f"{perf:.4f}" if perf is not None else "N/A")

    with col4:
        training_time = node.get("training_time")
        st.metric("Time", f"{training_time:.1f}s" if training_time is not None else "N/A")

    # Error details
    if node.get("is_buggy", False):
        error = node.get("error", "Unknown error")
        st.error(f"Error: {error}")
        return  # Don't show code for buggy solutions

    # Pipeline code
    st.subheader("Feature Engineering Code")

    pipeline_code_path = exp_path / DirNames.BUILD_DIR / "search" / "pipelines" / f"solution{solution_id}_pipeline.py"
    pipeline_code = load_code_file(pipeline_code_path)

    if pipeline_code:
        st.code(pipeline_code, language="python", line_numbers=True)
    else:
        st.info("Pipeline code not available")

    # Plan details in expander
    plan = node.get("plan") or {}
    if plan:
        with st.expander("Plan Details"):
            features = plan.get("features") or {}
            model = plan.get("model") or {}

            st.text(f"Feature Strategy: {features.get('strategy', 'N/A')}")
            st.text(f"Model Directive: {model.get('directive', 'N/A')}")
            st.text(f"Changes: {model.get('change_summary', 'N/A')}")
