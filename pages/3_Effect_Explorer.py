import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import ensure_demo_data, configured_data
from utils.transformations import (
    geometric_adstock,
    hill_saturation,
    transform_media,
    marginal_response,
    steady_state_adstock,
    effective_spend_range,
)
from utils.simulation import generate_channel_response_curve
from utils.plotting import get_channel_color

st.set_page_config(page_title="Effect Explorer | MMM Workbench", layout="wide")
ensure_demo_data()

df = configured_data()
channels = st.session_state.channel_cols

st.title("Effect Explorer")
st.caption("Understand how raw spend transforms into modeled media effects through adstock (carryover) and saturation (diminishing returns).")

with st.expander("How to use this page", expanded=False):
    st.markdown("""
    **Adstock (Carryover)**: Advertising doesn't just work in the period it runs—it lingers.
    The geometric adstock model assumes each period retains a fraction (`decay`) of the previous period's effect.
    - **Decay = 0**: No carryover (effect only in current period)
    - **Decay = 0.5**: Half the effect carries to next week
    - **Decay = 0.8**: Strong carryover (effect lasts ~5 weeks)

    **Saturation (Diminishing Returns)**: The first dollar spent is more effective than the hundredth.
    The Hill function models this S-shaped curve:
    - **Strength**: How abruptly the curve bends (higher = sharper saturation)
    - **Midpoint**: Spend level where response reaches 50% of maximum

    **Marginal Response**: The derivative of the saturation curve—how much *additional* outcome
    each *additional* dollar generates. This is what matters for budget optimization.
    """)

if not channels:
    st.warning("No channels configured. Go to **Data** page to select media spend columns.")
    st.stop()

# Channel selector
col_channel, col_preset = st.columns([3, 1])
with col_channel:
    channel = st.selectbox("Select channel to explore", channels, key="effect_channel")
with col_preset:
    preset = st.selectbox(
        "Preset profile",
        ["Custom", "Search (fast decay, moderate saturation)", "Video (slow decay, low saturation)", "Display (fast decay, high saturation)"],
        key="effect_preset"
    )

# Preset parameters
presets = {
    "Search (fast decay, moderate saturation)": {"decay": 0.58, "strength": 1.35, "midpoint_scale": 1.0},
    "Video (slow decay, low saturation)": {"decay": 0.72, "strength": 1.2, "midpoint_scale": 1.0},
    "Display (fast decay, high saturation)": {"decay": 0.30, "strength": 1.9, "midpoint_scale": 1.0},
}

if preset != "Custom":
    p = presets[preset]
    default_decay = p["decay"]
    default_strength = p["strength"]
else:
    default_decay = 0.5
    default_strength = 1.5

raw = df[channel].clip(lower=0).to_numpy(float)
default_mid = float(np.median(raw[raw > 0])) if np.any(raw > 0) else 1.0
max_raw = float(raw.max())

# Parameter controls
st.subheader("Transformation Parameters")
c1, c2, c3, c4 = st.columns(4)
decay = c1.slider("Adstock decay (λ)", 0.0, 0.95, default_decay, 0.01, key="decay_slider",
                  help="Fraction of effect retained each period. Higher = longer carryover.")
l_max = c2.slider("Carryover window (periods)", 1, 20, 8, 1, key="lmax_slider",
                  help="Maximum lag periods to include in adstock sum.")
strength = c3.slider("Saturation strength (Hill α)", 0.3, 3.0, default_strength, 0.05, key="strength_slider",
                     help="Steepness of saturation curve. Higher = more abrupt diminishing returns.")
midpoint = c4.slider("Saturation midpoint", 1.0, max(max_raw * 2, 2.0), default_mid, 100.0, key="midpoint_slider",
                     help="Spend level where response reaches 50% of max.")

# Compute transformations
adstocked = geometric_adstock(raw, decay, l_max)
transformed = hill_saturation(adstocked, strength, midpoint)
marginal = marginal_response(raw, decay, l_max, strength, midpoint)

# Steady-state curve for visualization
x_curve = np.linspace(0, max(max_raw * 1.5, midpoint * 2), 200)
adstocked_curve = x_curve / (1 - decay + 1e-6)
saturated_curve = hill_saturation(adstocked_curve, strength, midpoint)
marginal_curve = marginal_response(x_curve, decay, l_max, strength, midpoint)

# Effective carryover info
eff_periods = effective_spend_range(decay, l_max)

# Channel color for consistent visual identity
channel_color = get_channel_color(channel)

# --- Chart 1: Raw vs Adstocked Spend Over Time ---
st.subheader("1. Adstock: Spend Over Time with Carryover")
fig1 = make_subplots(specs=[[{"secondary_y": False}]])
fig1.add_trace(
    go.Scatter(
        x=df[st.session_state.date_col], y=raw,
        name="Raw spend", line=dict(color="#94A3B8", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Raw: %{y:,.0f}<extra></extra>"
    )
)
fig1.add_trace(
    go.Scatter(
        x=df[st.session_state.date_col], y=adstocked,
        name="Adstocked spend", line=dict(color=channel_color, width=3),
        hovertemplate="%{x|%b %d, %Y}<br>Adstocked: %{y:,.0f}<extra></extra>"
    )
)
fig1.update_layout(
    title=f"{channel.title()}: Raw vs Adstocked Spend",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=50, b=10),
    yaxis_title="Spend",
)
st.plotly_chart(fig1, use_container_width=True)

# Interpretation
c1, c2 = st.columns(2)
with c1:
    st.info(f"""
    **Adstock Effect (λ={decay:.2f}, window={l_max})**
    - Each period retains **{decay:.0%}** of previous period's effect
    - Effective carryover: **{eff_periods} periods** (until effect < 1%)
    - Total multiplier: **1/(1-λ) = {1/(1-decay+1e-6):.1f}x** at steady state
    - Adstocked spend = raw spend + λ·raw_spend(t-1) + λ²·raw_spend(t-2) + ...
    """)
with c2:
    # Adstock weight visualization
    weights = decay ** np.arange(l_max + 1)
    fig_w = go.Figure(go.Bar(x=list(range(l_max + 1)), y=weights, marker_color=channel_color))
    fig_w.update_layout(
        title="Adstock Weights by Lag",
        template="plotly_white",
        xaxis_title="Lag (periods ago)",
        yaxis_title="Weight",
        margin=dict(l=10, r=10, t=40, b=10),
        height=250,
    )
    st.plotly_chart(fig_w, use_container_width=True)

# --- Chart 2: Saturation Curve ---
st.subheader("2. Saturation: Diminishing Returns Curve")
c1, c2 = st.columns(2)

with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_curve, y=saturated_curve,
        fill="tozeroy", fillcolor=f"rgba{tuple(list(bytes.fromhex(channel_color[1:])) + [0.15])}",
        line=dict(color=channel_color, width=3),
        name="Response",
        hovertemplate="Spend: %{x:,.0f}<br>Response: %{y:.3f}<extra></extra>"
    ))
    # Mark current average spend
    avg_spend = raw.mean()
    avg_adstocked = steady_state_adstock(avg_spend, decay)
    avg_response = hill_saturation(avg_adstocked, strength, midpoint)
    fig2.add_trace(go.Scatter(
        x=[avg_spend], y=[avg_response],
        mode="markers", marker=dict(color="#F59E0B", size=12, symbol="diamond"),
        name=f"Current avg (${avg_spend:,.0f})",
        hovertemplate="Current avg spend: %{x:,.0f}<br>Response: %{y:.3f}<extra></extra>"
    ))
    fig2.add_vline(x=midpoint, line_dash="dash", line_color="#EF4444", opacity=0.5,
                   annotation_text=f"Midpoint (${midpoint:,.0f})", annotation_position="top")
    fig2.update_layout(
        title="Saturation Response Curve",
        template="plotly_white",
        xaxis_title="Average-period raw spend",
        yaxis_title="Normalized response (0–1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=x_curve, y=marginal_curve,
        line=dict(color="#F59E0B", width=3),
        fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.15)",
        name="Marginal response",
        hovertemplate="Spend: %{x:,.0f}<br>Marginal: %{y:.4f}<extra></extra>"
    ))
    fig3.add_trace(go.Scatter(
        x=[avg_spend], y=[marginal_response(np.array([avg_spend]), decay, l_max, strength, midpoint)[0]],
        mode="markers", marker=dict(color="#EF4444", size=12, symbol="diamond"),
        name=f"Current marginal",
        hovertemplate="Marginal at current: %{y:.4f}<extra></extra>"
    ))
    fig3.update_layout(
        title="Marginal Response (Derivative)",
        template="plotly_white",
        xaxis_title="Average-period raw spend",
        yaxis_title="Incremental response per $1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# Interpretation
st.info(f"""
**Saturation Effect (α={strength:.2f}, midpoint=${midpoint:,.0f})**
- At current avg spend (${avg_spend:,.0f}): response = **{avg_response:.3f}** (of max 1.0), marginal = **{marginal_response(np.array([avg_spend]), decay, l_max, strength, midpoint)[0]:.4f}**
- Midpoint (${midpoint:,.0f}): half-maximum response, marginal = **{marginal_response(np.array([midpoint]), decay, l_max, strength, midpoint)[0]:.4f}** (peak marginal for α>1)
- Strength {strength:.2f}: {'gradual' if strength < 1.2 else 'moderate' if strength < 1.8 else 'sharp'} diminishing returns
""")

# --- Chart 3: Full Transformation Pipeline ---
st.subheader("3. Complete Pipeline: Raw Spend to Modeled Effect")
fig4 = make_subplots(rows=1, cols=3, subplot_titles=("Raw Spend", "Adstocked", "Saturated (Model Input)"),
                     horizontal_spacing=0.08)

fig4.add_trace(go.Histogram(x=raw, nbinsx=30, name="Raw", marker_color="#94A3B8", showlegend=False), row=1, col=1)
fig4.add_trace(go.Histogram(x=adstocked, nbinsx=30, name="Adstocked", marker_color=channel_color, showlegend=False), row=1, col=2)
fig4.add_trace(go.Histogram(x=transformed, nbinsx=30, name="Saturated", marker_color="#4C72B0", showlegend=False), row=1, col=3)

fig4.update_layout(
    template="plotly_white",
    title="Distribution Transformation",
    margin=dict(l=10, r=10, t=50, b=10),
    height=350,
)
st.plotly_chart(fig4, use_container_width=True)

# Summary stats
st.subheader("Summary Statistics")
stats_df = pd.DataFrame({
    "Metric": ["Mean", "Std Dev", "Min", "Max", "Skewness"],
    "Raw Spend": [f"{raw.mean():,.0f}", f"{raw.std():,.0f}", f"{raw.min():,.0f}", f"{raw.max():,.0f}", f"{pd.Series(raw).skew():.2f}"],
    "Adstocked": [f"{adstocked.mean():,.0f}", f"{adstocked.std():,.0f}", f"{adstocked.min():,.0f}", f"{adstocked.max():,.0f}", f"{pd.Series(adstocked).skew():.2f}"],
    "Saturated": [f"{transformed.mean():.3f}", f"{transformed.std():.3f}", f"{transformed.min():.3f}", f"{transformed.max():.3f}", f"{pd.Series(transformed).skew():.2f}"],
})
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# Theoretical response curve (if model exists)
if "model_result" in st.session_state and st.session_state.model_result:
    result = st.session_state.model_result
    if channel in result["channels"]:
        st.subheader("4. Modeled Response Curve (from fitted model)")
        from utils.optimization import response_curve_data
        spend_vals, mean_resp, low_resp, high_resp = response_curve_data(channel, result)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=np.concatenate([spend_vals, spend_vals[::-1]]),
            y=np.concatenate([high_resp, low_resp[::-1]]),
            fill="toself", fillcolor="rgba(76, 114, 176, 0.15)",
            line=dict(width=0), name="90% credible interval", hoverinfo="skip"
        ))
        fig5.add_trace(go.Scatter(
            x=spend_vals, y=mean_resp,
            line=dict(color="#4C72B0", width=3), name="Posterior mean"
        ))
        current_spend = result["current_spend"][channel]
        fig5.add_vline(x=current_spend, line_dash="dash", line_color="#F59E0B",
                       annotation_text=f"Current (${current_spend:,.0f})", annotation_position="top")
        fig5.update_layout(
            title=f"{channel.title()}: Expected Incremental Outcome vs Spend",
            template="plotly_white",
            xaxis_title="Average-period spend",
            yaxis_title="Expected incremental outcome",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)