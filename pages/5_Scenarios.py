import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import ensure_demo_data
from utils import _lazy_import_modeling

st.set_page_config(page_title="Scenarios | MMM Workbench", page_icon=":material/target:", layout="wide")
ensure_demo_data()

# Lazy imports
_m = _lazy_import_modeling()
scenario_lift = _m["scenario_lift"]

result = st.session_state.get("model_result")

st.title("Scenario Planner")
st.caption("Test hypothetical spend changes against the fitted baseline with full posterior uncertainty.")

with st.expander(":material/info: How scenarios work", expanded=False):
    st.markdown("""
    **What this does:**
    - Takes your fitted Bayesian model (with all its uncertainty)
    - Computes expected incremental outcome for a new spend plan vs. current average spend
    - Uses the *same* adstock/saturation transformations from the model
    - Returns a full posterior distribution of lift (not just a point estimate)
    
    **Key concepts:**
    - **Baseline**: Current average-period spend per channel (from fitted data)
    - **Scenario**: Your proposed average-period spend per channel
    - **Lift**: Scenario outcome - Baseline outcome (incremental)
    - **Credible interval**: 90% posterior interval — if it spans zero, the effect is uncertain
    
    **Common questions to answer:**
    - "What if we increase Search by 20%?"
    - "What happens if we cut Display by 30% and reinvest in Video?"
    - "Can we maintain sales with 10% less total budget?"
    - "What's the incremental ROAS of an extra $10k in Social?"
    """)

if not result:
    st.warning("No fitted model found. Go to **Model** page and fit a Bayesian MMM first.")
    st.stop()

channels = result["channels"]
current_spend = result["current_spend"]

# Scenario configuration
st.subheader("Define Your Scenario")

# Quick preset buttons
st.markdown("**Quick presets:**")
preset_cols = st.columns(5)
presets = {
    "Status Quo": {c: 0 for c in channels},
    "+10% All": {c: 10 for c in channels},
    "-10% All": {c: -10 for c in channels},
    "Shift to Video": {c: 20 if c == "video" else -10 for c in channels},
    "Cut Display": {c: -50 if c == "display" else 0 for c in channels},
}

for i, (name, changes) in enumerate(presets.items()):
    if preset_cols[i].button(name, use_container_width=True, key=f"preset_{name}"):
        for c, chg in changes.items():
            st.session_state[f"scenario_{c}"] = chg
        st.rerun()

# Channel sliders
st.markdown("**Adjust per-channel spend (% change from baseline):**")
plan = {}
cols = st.columns(len(channels))

for col, channel in zip(cols, channels):
    baseline = current_spend[channel]
    default_change = st.session_state.get(f"scenario_{channel}", 0)
    change = col.slider(
        f"{channel.title()}",
        -100, 200, default_change, 5,
        key=f"scenario_{channel}",
        help=f"Baseline: ${baseline:,.0f}/period"
    )
    plan[channel] = baseline * (1 + change / 100)
    col.caption(f"Baseline: ${baseline:,.0f} → Plan: ${plan[channel]:,.0f}")

# Ensure plan has all required channels
for c in channels:
    if c not in plan:
        plan[c] = current_spend[c]

# Budget constraint option
st.markdown("---")
budget_mode = st.radio(
    "Budget constraint",
    ["Flexible total budget", "Fixed total budget (rebalance)"],
    horizontal=True,
    help="Flexible: sum of plan can differ from baseline. Fixed: total stays constant, channels rebalanced."
)

if budget_mode == "Fixed total budget (rebalance)":
    current_total = sum(current_spend.values())
    plan_total = sum(plan.values())
    if abs(plan_total - current_total) > 1:
        # Rebalance proportionally to maintain total
        scale = current_total / plan_total
        plan = {c: v * scale for c, v in plan.items()}
        st.info(f"Plan rebalanced to match baseline total: ${current_total:,.0f}/period")
        # Show adjusted values
        for c in channels:
            st.caption(f"{c.title()}: ${plan[c]:,.0f} (adjusted)")

# Compute lift
lift = scenario_lift(plan, result)
mean_lift = lift.mean()
low_lift = np.quantile(lift, 0.05)
high_lift = np.quantile(lift, 0.95)
median_lift = np.median(lift)
prob_positive = (lift > 0).mean()

# Results summary
st.subheader("Scenario Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Incremental Outcome", f"{mean_lift:,.0f}")
c2.metric("90% Credible Interval", f"{low_lift:,.0f} to {high_lift:,.0f}")
c3.metric("Probability Positive", f"{prob_positive:.1%}")
c4.metric("Budget Change", f"{(sum(plan.values())/sum(current_spend.values())-1)*100:+.1f}%")

# Interpretation
if low_lift < 0 < high_lift:
    st.warning(f"""
    ⚠️ **Uncertain impact**: The 90% credible interval spans zero ({low_lift:,.0f} to {high_lift:,.0f}).
    Probability of positive lift: **{prob_positive:.1%}**.
    This scenario's effect is not statistically distinguishable from zero given current data and model uncertainty.
    """)
elif low_lift > 0:
    st.success(f"""
    ✅ **Positive impact likely**: The 90% credible interval is entirely positive ({low_lift:,.0f} to {high_lift:,.0f}).
    Probability of positive lift: **{prob_positive:.1%}**.
    The modeled effect is directionally consistent; validate with experiments where possible.
    """)
else:
    st.error(f"""
    ❌ **Negative impact likely**: The 90% credible interval is entirely negative ({low_lift:,.0f} to {high_lift:,.0f}).
    Probability of positive lift: **{prob_positive:.1%}**.
    This scenario is modeled to decrease the outcome.
    """)

# --- Visualization 1: Spend Comparison ---
st.subheader("Spend Plan vs Baseline")
compare_df = pd.DataFrame({
    "Channel": channels,
    "Baseline": [current_spend[c] for c in channels],
    "Scenario": [plan[c] for c in channels],
})
compare_df["Change"] = compare_df["Scenario"] - compare_df["Baseline"]
compare_df["Change %"] = (compare_df["Change"] / compare_df["Baseline"] * 100).round(1)

fig_spend = go.Figure()
fig_spend.add_trace(go.Bar(
    name="Baseline", x=compare_df["Channel"], y=compare_df["Baseline"],
    marker_color="#94A3B8", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
))
fig_spend.add_trace(go.Bar(
    name="Scenario", x=compare_df["Channel"], y=compare_df["Scenario"],
    marker_color="#0F766E", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
))
fig_spend.update_layout(
    barmode="group", template="plotly_white",
    title="Average-Period Spend: Baseline vs Scenario",
    yaxis_title="Spend", legend=dict(orientation="h", y=1.02),
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_spend, use_container_width=True)

# Change table
compare_df_fmt = compare_df.copy()
compare_df_fmt["Baseline"] = compare_df_fmt["Baseline"].apply(lambda x: f"${x:,.0f}")
compare_df_fmt["Scenario"] = compare_df_fmt["Scenario"].apply(lambda x: f"${x:,.0f}")
compare_df_fmt["Change"] = compare_df_fmt["Change"].apply(lambda x: f"${x:+,.0f}")
compare_df_fmt["Change %"] = compare_df_fmt["Change %"].apply(lambda x: f"{x:+.1f}%")
st.dataframe(
    compare_df_fmt,
    use_container_width=True, hide_index=True
)

# --- Visualization 2: Lift Distribution ---
st.subheader("Incremental Lift Distribution")
fig_lift = go.Figure()
fig_lift.add_trace(go.Histogram(
    x=lift, nbinsx=50, name="Posterior lift",
    marker_color="#4C72B0", opacity=0.7,
    hovertemplate="Lift: %{x:,.0f}<br>Count: %{y}<extra></extra>"
))
fig_lift.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero")
fig_lift.add_vline(x=mean_lift, line_dash="solid", line_color="#EF4444",
                   annotation_text=f"Mean: {mean_lift:,.0f}", annotation_position="top")
fig_lift.add_vline(x=low_lift, line_dash="dot", line_color="#F59E0B",
                   annotation_text=f"5%: {low_lift:,.0f}", annotation_position="top")
fig_lift.add_vline(x=high_lift, line_dash="dot", line_color="#F59E0B",
                   annotation_text=f"95%: {high_lift:,.0f}", annotation_position="top")
fig_lift.update_layout(
    template="plotly_white",
    title="Posterior Distribution of Incremental Outcome",
    xaxis_title="Incremental outcome (scenario - baseline)",
    yaxis_title="Posterior density (count)",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_lift, use_container_width=True)

# --- Visualization 3: Channel-level Decomposition ---
st.subheader("Channel-Level Lift Decomposition")
channel_lifts = {}
for c in channels:
    baseline_resp = np.zeros_like(lift)
    scenario_resp = np.zeros_like(lift)
    # We need to recompute per-channel
    from utils.modeling import response_samples
    baseline_resp += response_samples(current_spend[c], c, result, len(lift))
    scenario_resp += response_samples(plan[c], c, result, len(lift))
    channel_lifts[c] = scenario_resp - baseline_resp

decomp_df = pd.DataFrame({
    "Channel": channels,
    "Mean Lift": [channel_lifts[c].mean() for c in channels],
    "5%": [np.quantile(channel_lifts[c], 0.05) for c in channels],
    "95%": [np.quantile(channel_lifts[c], 0.95) for c in channels],
    "P(Positive)": [(channel_lifts[c] > 0).mean() for c in channels],
})

fig_decomp = go.Figure()
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
for i, row in decomp_df.iterrows():
    fig_decomp.add_trace(go.Bar(
        name=row["Channel"], x=[row["Channel"]], y=[row["Mean Lift"]],
        marker_color=colors[i % len(colors)],
        error_y=dict(
            type="data", symmetric=False,
            array=[row["95%"] - row["Mean Lift"]],
            arrayminus=[row["Mean Lift"] - row["5%"]],
            color="gray"
        ),
        hovertemplate=f"{row['Channel']}<br>Mean: %{{y:,.0f}}<br>90% CI: [{row['5%']:,.0f}, {row['95%']:,.0f}]<extra></extra>"
    ))
fig_decomp.add_hline(y=0, line_dash="dash", line_color="gray")
fig_decomp.update_layout(
    template="plotly_white", barmode="group",
    title="Incremental Lift by Channel (with 90% CI)",
    yaxis_title="Incremental outcome", showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_decomp, use_container_width=True)

decomp_df_fmt = decomp_df.copy()
decomp_df_fmt["Mean Lift"] = decomp_df_fmt["Mean Lift"].apply(lambda x: f"{x:,.0f}")
decomp_df_fmt["5%"] = decomp_df_fmt["5%"].apply(lambda x: f"{x:,.0f}")
decomp_df_fmt["95%"] = decomp_df_fmt["95%"].apply(lambda x: f"{x:,.0f}")
decomp_df_fmt["P(Positive)"] = decomp_df_fmt["P(Positive)"].apply(lambda x: f"{x:.1%}")
st.dataframe(
    decomp_df_fmt,
    use_container_width=True, hide_index=True
)

# --- Visualization 4: Response Curves with Scenario Points ---
st.subheader("Response Curves with Scenario Position")
from utils.optimization import response_curve_data

fig_curves = make_subplots(
    rows=1, cols=len(channels),
    subplot_titles=[c.title() for c in channels],
    horizontal_spacing=0.05
)

for i, c in enumerate(channels):
    spend_vals, mean_resp, low_resp, high_resp = response_curve_data(c, result)
    fig_curves.add_trace(go.Scatter(
        x=np.concatenate([spend_vals, spend_vals[::-1]]),
        y=np.concatenate([high_resp, low_resp[::-1]]),
        fill="toself", fillcolor=f"rgba{tuple(list(bytes.fromhex(colors[i][1:])) + [0.15])}",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ), row=1, col=i+1)
    fig_curves.add_trace(go.Scatter(
        x=spend_vals, y=mean_resp,
        line=dict(color=colors[i], width=2), showlegend=False
    ), row=1, col=i+1)
    # Baseline and scenario points
    fig_curves.add_trace(go.Scatter(
        x=[current_spend[c]], y=[np.interp(current_spend[c], spend_vals, mean_resp)],
        mode="markers", marker=dict(color="#94A3B8", size=12, symbol="circle"),
        name="Baseline" if i == 0 else None, showlegend=(i == 0),
        hovertemplate=f"Baseline: {current_spend[c]:,.0f}<br>Response: %{{y:,.0f}}<extra></extra>"
    ), row=1, col=i+1)
    fig_curves.add_trace(go.Scatter(
        x=[plan[c]], y=[np.interp(plan[c], spend_vals, mean_resp)],
        mode="markers", marker=dict(color="#EF4444", size=12, symbol="diamond"),
        name="Scenario" if i == 0 else None, showlegend=(i == 0),
        hovertemplate=f"Scenario: {plan[c]:,.0f}<br>Response: %{{y:,.0f}}<extra></extra>"
    ), row=1, col=i+1)

fig_curves.update_layout(
    template="plotly_white", height=400,
    title="Channel Response Curves: Baseline (●) vs Scenario (◆)",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", y=1.02),
)
fig_curves.update_xaxes(title_text="Spend")
fig_curves.update_yaxes(title_text="Incremental Outcome")
st.plotly_chart(fig_curves, use_container_width=True)

# --- Save scenario ---
if st.button("Save scenario for comparison", type="secondary"):
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = []
    st.session_state.saved_scenarios.append({
        "name": f"Scenario {len(st.session_state.saved_scenarios) + 1}",
        "plan": plan.copy(),
        "lift_mean": mean_lift,
        "lift_low": low_lift,
        "lift_high": high_lift,
    })
    st.success("Scenario saved!")

# Show saved scenarios
    if "saved_scenarios" in st.session_state and st.session_state.saved_scenarios:
        st.subheader("Saved Scenarios Comparison")
        saved_df = pd.DataFrame(st.session_state.saved_scenarios)
        saved_df_fmt = saved_df[["name", "lift_mean", "lift_low", "lift_high"]].copy()
        saved_df_fmt["lift_mean"] = saved_df_fmt["lift_mean"].apply(lambda x: f"{x:,.0f}")
        saved_df_fmt["lift_low"] = saved_df_fmt["lift_low"].apply(lambda x: f"{x:,.0f}")
        saved_df_fmt["lift_high"] = saved_df_fmt["lift_high"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(
            saved_df_fmt,
            use_container_width=True, hide_index=True
        )