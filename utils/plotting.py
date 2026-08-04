"""Plotting utilities for MMM Workbench."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def line_with_band(
    df: pd.DataFrame,
    x: str,
    mean: str,
    low: str,
    high: str,
    title: str,
    observed: str | None = None,
) -> go.Figure:
    """Create a line chart with credible interval band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[high], mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df[x], y=df[low], mode="lines", fill="tonexty",
        fillcolor="rgba(76, 114, 176, 0.18)", line=dict(width=0),
        name="90% credible interval"
    ))
    fig.add_trace(go.Scatter(
        x=df[x], y=df[mean], mode="lines",
        line=dict(color="#4C72B0", width=3), name="Posterior mean"
    ))
    if observed and observed in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[observed], mode="markers",
            marker=dict(color="#1F2937", size=5, opacity=0.6), name="Observed"
        ))
    fig.update_layout(
        title=title, template="plotly_white",
        legend_orientation="h", legend=dict(y=1.02),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified"
    )
    return fig


@st.cache_data
def allocation_chart(current: dict, allocation: dict) -> go.Figure:
    """Create grouped bar chart comparing current vs recommended allocation."""
    df = pd.DataFrame({
        "channel": list(allocation),
        "Current": [current[c] for c in allocation],
        "Recommended": list(allocation.values()),
    }).melt("channel", var_name="Plan", value_name="Spend")
    return px.bar(
        df, x="channel", y="Spend", color="Plan", barmode="group",
        template="plotly_white",
        title="Current vs Recommended Average-Period Allocation",
        color_discrete_sequence=["#94A3B8", "#0F766E"]
    )


@st.cache_data
def response_curve_plot(
    spend_vals, mean_resp, low_resp, high_resp,
    current_spend, optimal_spend, channel_name, color
) -> go.Figure:
    """Create a single response curve plot with current/optimal markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(spend_vals) + list(spend_vals[::-1]),
        y=list(high_resp) + list(low_resp[::-1]),
        fill="toself", fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.15])}",
        line=dict(width=0), name="90% CI", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=spend_vals, y=mean_resp, line=dict(color=color, width=2),
        name="Posterior mean", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[current_spend], y=[__import__("numpy").interp(current_spend, spend_vals, mean_resp)],
        mode="markers", marker=dict(color="#94A3B8", size=12, symbol="circle", line=dict(width=2, color="white")),
        name="Current", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[optimal_spend], y=[__import__("numpy").interp(optimal_spend, spend_vals, mean_resp)],
        mode="markers", marker=dict(color="#EF4444", size=12, symbol="diamond", line=dict(width=2, color="white")),
        name="Optimal", showlegend=True
    ))
    fig.update_layout(
        template="plotly_white", title=channel_name.title(),
        xaxis_title="Spend", yaxis_title="Incremental Outcome",
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
    )
    return fig


@st.cache_data
def posterior_density_plot(draws_dict, colors, title="Posterior Distributions"):
    """Create overlay density plots for multiple parameter posteriors."""
    fig = go.Figure()
    for i, (name, draws) in enumerate(draws_dict.items()):
        hist, bin_edges = __import__("numpy").histogram(draws, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=bin_centers, y=hist,
            fill="tozeroy", fillcolor=f"rgba{tuple(list(bytes.fromhex(c[1:])) + [0.3])}",
            line=dict(color=c, width=2), name=name,
            hovertemplate=f"{name}<br>Value: %{{x:.3f}}<br>Density: %{{y:.2f}}<extra></extra>"
        ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_white", title=title,
        xaxis_title="Parameter value", yaxis_title="Density",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.02)
    )
    return fig