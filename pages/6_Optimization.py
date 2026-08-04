import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import ensure_demo_data
from utils.optimization import optimize_budget, compute_channel_roi, response_curve_data

st.set_page_config(page_title="Optimization | MMM Workbench", page_icon="⚙️", layout="wide")
ensure_demo_data()

result = st.session_state.get("model_result")

st.title("Budget Optimizer")
st.caption("Find the optimal budget allocation across channels to maximize expected outcome under constraints.")

with st.expander("📖 How optimization works", expanded=False):
    st.markdown("""
    **Objective:** Maximize total expected incremental outcome
    $$\\max_{x} \\sum_c \\mathbb{E}[f_c(x_c)] \\quad \\text{s.t.} \\quad \\sum_c x_c = B, \\quad L_c \\le x_c \\le U_c$$

    **Where:**
    - $x_c$ = average-period spend for channel $c$
    - $f_c$ = channel response curve (adstock + saturation, with posterior uncertainty)
    - $B$ = total budget constraint
    - $L_c, U_c$ = minimum/maximum spend per channel
    
    **Method:**
    - Uses posterior mean response curves for optimization (fast)
    - Reports full posterior lift distribution vs. current allocation
    - SLSQP optimizer with bound and equality constraints
    
    **Assumptions & Caveats:**
    - Response curves estimated from historical data; extrapolation beyond observed spend range is uncertain
    - Assumes channel independence (no interaction effects)
    - Uses steady-state adstock approximation (valid for sustained spend levels)
    - Optimizes *average-period* spend; actual weekly allocation may vary
    - **Always validate recommendations with experiments before committing budget**
    """)

if not result:
    st.warning("No fitted model found. Go to **Model** page and fit a Bayesian MMM first.")
    st.stop()

channels = result["channels"]
current_spend = result["current_spend"]
current_total = sum(current_spend.values())

# Budget input
st.subheader("Budget & Constraints")
c1, c2 = st.columns([1, 3])
with c1:
    total_budget = st.number_input(
        "Total average-period budget",
        min_value=100.0, value=float(current_total), step=1000.0,
        help="Total budget to allocate across all channels (per period average)"
    )

# Constraint presets
st.markdown("**Constraint presets:**")
preset_cols = st.columns(4)
presets = {
    "Current ±25%": (0.25, 3.0),
    "Current ±50%": (0.5, 2.0),
    "Wide (10%-10x)": (0.1, 10.0),
    "Tight (±10%)": (0.9, 1.1),
}

for i, (name, (min_mult, max_mult)) in enumerate(presets.items()):
    if preset_cols[i].button(name, use_container_width=True, key=f"opt_preset_{name}"):
        for c in channels:
            st.session_state[f"min_{c}"] = current_spend[c] * min_mult
            st.session_state[f"max_{c}"] = current_spend[c] * max_mult
        st.rerun()

# Per-channel constraints
st.markdown("**Per-channel constraints (average-period spend):**")
minimums = {}
maximums = {}

constraint_cols = st.columns(len(channels))
for col, channel in zip(constraint_cols, channels):
    current = current_spend[channel]
    default_min = st.session_state.get(f"min_{channel}", current * 0.25)
    default_max = st.session_state.get(f"max_{channel}", current * 3.0)
    
    with col:
        st.write(f"**{channel.title()}** (current: ${current:,.0f})")
        min_val = st.number_input(
            f"Min {channel}", 0.0, value=float(default_min), step=500.0,
            key=f"min_{channel}", help=f"Minimum spend for {channel}"
        )
        max_val = st.number_input(
            f"Max {channel}", 0.0, value=float(default_max), step=500.0,
            key=f"max_{channel}", help=f"Maximum spend for {channel}"
        )
        minimums[channel] = min_val
        maximums[channel] = max_val

# Validation
min_sum = sum(minimums.values())
max_sum = sum(maximums.values())
if min_sum > total_budget:
    st.error(f"❌ Sum of minimums (${min_sum:,.0f}) exceeds total budget (${total_budget:,.0f})")
elif max_sum < total_budget:
    st.error(f"❌ Sum of maximums (${max_sum:,.0f}) is less than total budget (${total_budget:,.0f})")
else:
    st.success(f"✅ Constraints feasible: budget ${total_budget:,.0f} ∈ [${min_sum:,.0f}, ${max_sum:,.0f}]")

# Optimization button
if st.button("Find Recommended Allocation", type="primary", use_container_width=True, disabled=(min_sum > total_budget or max_sum < total_budget)):
    with st.spinner("Optimizing budget allocation..."):
        try:
            allocation, lift_samples = optimize_budget(
                total_budget, minimums, maximums, result, n_draws=300
            )
            st.session_state.optimization = {
                "allocation": allocation,
                "lift_samples": lift_samples,
                "total_budget": total_budget,
                "minimums": minimums,
                "maximums": maximums,
            }
            st.success("Optimization complete!")
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
            st.exception(exc)

opt = st.session_state.get("optimization")
if not opt:
    st.info("Set budget and constraints above, then click **Find Recommended Allocation**.")
    st.stop()

allocation = opt["allocation"]
lift_samples = opt["lift_samples"]

# --- Results Summary ---
st.subheader("Optimization Results")

mean_lift = lift_samples.mean()
low_lift = np.quantile(lift_samples, 0.05)
high_lift = np.quantile(lift_samples, 0.95)
prob_pos = (lift_samples > 0).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Improvement", f"{mean_lift:,.0f}")
c2.metric("90% Credible Interval", f"{low_lift:,.0f} to {high_lift:,.0f}")
c3.metric("Probability Positive", f"{prob_pos:.1%}")
c4.metric("Recommended Budget", f"{sum(allocation.values()):,.0f}")

# --- Allocation Comparison Table ---
st.subheader("Allocation: Current vs Recommended")
table_data = []
for c in channels:
    cur = current_spend[c]
    rec = allocation[c]
    chg = rec - cur
    pct = (chg / cur * 100) if cur > 0 else 0
    table_data.append({
        "Channel": c.title(),
        "Current": cur,
        "Recommended": rec,
        "Change ($)": chg,
        "Change (%)": pct,
    })

alloc_df = pd.DataFrame(table_data)
alloc_df_fmt = alloc_df.copy()
alloc_df_fmt["Current"] = alloc_df_fmt["Current"].apply(lambda x: f"${x:,.0f}")
alloc_df_fmt["Recommended"] = alloc_df_fmt["Recommended"].apply(lambda x: f"${x:,.0f}")
alloc_df_fmt["Change ($)"] = alloc_df_fmt["Change ($)"].apply(lambda x: f"${x:+,.0f}")
alloc_df_fmt["Change (%)"] = alloc_df_fmt["Change (%)"].apply(lambda x: f"{x:+.1f}%")
st.dataframe(
    alloc_df_fmt,
    use_container_width=True, hide_index=True
)

# --- Allocation Bar Chart ---
fig_alloc = go.Figure()
fig_alloc.add_trace(go.Bar(
    name="Current", x=alloc_df["Channel"], y=alloc_df["Current"],
    marker_color="#94A3B8", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
))
fig_alloc.add_trace(go.Bar(
    name="Recommended", x=alloc_df["Channel"], y=alloc_df["Recommended"],
    marker_color="#0F766E", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
))
fig_alloc.update_layout(
    barmode="group", template="plotly_white",
    title="Current vs Recommended Average-Period Allocation",
    yaxis_title="Spend", legend=dict(orientation="h", y=1.02),
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_alloc, use_container_width=True)

# --- ROI Analysis ---
st.subheader("Channel ROI at Recommended Allocation")
roi_df = compute_channel_roi(allocation, result, n_draws=300)

# Add current ROI for comparison
roi_current = compute_channel_roi(current_spend, result, n_draws=300)
roi_df["Current_ROI"] = roi_current["ROI"].values
roi_df["Current_Marginal_ROAS"] = roi_current["Marginal_ROAS"].values
roi_df["Current_Spend"] = roi_current["Spend"].values

roi_df_fmt = roi_df.copy()
roi_df_fmt["Spend"] = roi_df_fmt["Spend"].apply(lambda x: f"${x:,.0f}")
roi_df_fmt["Expected_Response"] = roi_df_fmt["Expected_Response"].apply(lambda x: f"{x:,.0f}")
roi_df_fmt["ROI"] = roi_df_fmt["ROI"].apply(lambda x: f"{x:.2f}")
roi_df_fmt["Marginal_ROAS"] = roi_df_fmt["Marginal_ROAS"].apply(lambda x: f"{x:.2f}")
roi_df_fmt["Current_ROI"] = roi_df_fmt["Current_ROI"].apply(lambda x: f"{x:.2f}")
roi_df_fmt["Current_Marginal_ROAS"] = roi_df_fmt["Current_Marginal_ROAS"].apply(lambda x: f"{x:.2f}")
roi_df_fmt["Current_Spend"] = roi_df_fmt["Current_Spend"].apply(lambda x: f"${x:,.0f}")
st.dataframe(
    roi_df_fmt,
    use_container_width=True, hide_index=True
)

# ROI interpretation
st.markdown("""
**Interpretation:**
- **ROI** = Expected incremental outcome / Spend (average return)
- **Marginal ROAS** = Derivative of response curve at current spend (return on *next* dollar)
- Channels with high marginal ROAS relative to others are under-invested
- Optimization equalizes marginal ROAS across channels (subject to constraints)
""")

# --- Response Curves with Optimal Points ---
st.subheader("Response Curves: Current (●) vs Optimal (◆)")
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

fig_curves = make_subplots(
    rows=1, cols=len(channels),
    subplot_titles=[c.title() for c in channels],
    horizontal_spacing=0.05
)

for i, c in enumerate(channels):
    spend_vals, mean_resp, low_resp, high_resp = response_curve_data(c, result, max_multiplier=4.0)
    
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
    
    # Current point
    cur_spend = current_spend[c]
    cur_resp = np.interp(cur_spend, spend_vals, mean_resp)
    fig_curves.add_trace(go.Scatter(
        x=[cur_spend], y=[cur_resp],
        mode="markers", marker=dict(color="#94A3B8", size=14, symbol="circle", line=dict(width=2, color="white")),
        name="Current" if i == 0 else None, showlegend=(i == 0),
        hovertemplate=f"Current: {cur_spend:,.0f}<br>Response: %{{y:,.0f}}<extra></extra>"
    ), row=1, col=i+1)
    
    # Optimal point
    opt_spend = allocation[c]
    opt_resp = np.interp(opt_spend, spend_vals, mean_resp)
    fig_curves.add_trace(go.Scatter(
        x=[opt_spend], y=[opt_resp],
        mode="markers", marker=dict(color="#EF4444", size=14, symbol="diamond", line=dict(width=2, color="white")),
        name="Optimal" if i == 0 else None, showlegend=(i == 0),
        hovertemplate=f"Optimal: {opt_spend:,.0f}<br>Response: %{{y:,.0f}}<extra></extra>"
    ), row=1, col=i+1)
    
    # Constraint bounds
    fig_curves.add_vline(x=minimums[c], line_dash="dot", line_color="gray", opacity=0.5, row=1, col=i+1)
    fig_curves.add_vline(x=maximums[c], line_dash="dot", line_color="gray", opacity=0.5, row=1, col=i+1)

fig_curves.update_layout(
    template="plotly_white", height=400,
    title="Channel Response Curves with Constraint Bounds (dotted lines)",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", y=1.02),
)
fig_curves.update_xaxes(title_text="Spend")
fig_curves.update_yaxes(title_text="Incremental Outcome")
st.plotly_chart(fig_curves, use_container_width=True)

# --- Lift Distribution ---
st.subheader("Incremental Lift Distribution (Optimal vs Current)")
fig_lift = go.Figure()
fig_lift.add_trace(go.Histogram(
    x=lift_samples, nbinsx=50, name="Posterior lift",
    marker_color="#0F766E", opacity=0.7
))
fig_lift.add_vline(x=0, line_dash="dash", line_color="gray")
fig_lift.add_vline(x=mean_lift, line_dash="solid", line_color="#EF4444",
                   annotation_text=f"Mean: {mean_lift:,.0f}", annotation_position="top")
fig_lift.add_vline(x=low_lift, line_dash="dot", line_color="#F59E0B",
                   annotation_text=f"5%: {low_lift:,.0f}", annotation_position="top")
fig_lift.add_vline(x=high_lift, line_dash="dot", line_color="#F59E0B",
                   annotation_text=f"95%: {high_lift:,.0f}", annotation_position="top")
fig_lift.update_layout(
    template="plotly_white",
    title="Posterior Distribution of Incremental Improvement",
    xaxis_title="Incremental outcome (optimal - current)",
    yaxis_title="Count", margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_lift, use_container_width=True)

# --- Sensitivity: Budget Level ---
st.subheader("Sensitivity: Optimal Outcome vs Total Budget")
budget_range = np.linspace(min_sum, max_sum, 15)
sensitivity_results = []

for b in budget_range:
    try:
        alloc_b, lift_b = optimize_budget(b, minimums, maximums, result, n_draws=150)
        sensitivity_results.append({
            "Budget": b,
            "Mean_Lift": lift_b.mean(),
            "Low_Lift": np.quantile(lift_b, 0.05),
            "High_Lift": np.quantile(lift_b, 0.95),
        })
    except Exception:
        pass

if sensitivity_results:
    sens_df = pd.DataFrame(sensitivity_results)
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=np.concatenate([sens_df["Budget"], sens_df["Budget"][::-1]]),
        y=np.concatenate([sens_df["High_Lift"], sens_df["Low_Lift"][::-1]]),
        fill="toself", fillcolor="rgba(15, 118, 110, 0.15)", line=dict(width=0),
        name="90% CI", hoverinfo="skip"
    ))
    fig_sens.add_trace(go.Scatter(
        x=sens_df["Budget"], y=sens_df["Mean_Lift"],
        line=dict(color="#0F766E", width=3), name="Expected lift vs current"
    ))
    fig_sens.add_vline(x=current_total, line_dash="dash", line_color="#94A3B8",
                       annotation_text=f"Current (${current_total:,.0f})", annotation_position="top")
    fig_sens.add_vline(x=total_budget, line_dash="dash", line_color="#EF4444",
                       annotation_text=f"Selected (${total_budget:,.0f})", annotation_position="top")
    fig_sens.update_layout(
        template="plotly_white",
        title="Expected Incremental Outcome vs Total Budget (with optimal reallocation)",
        xaxis_title="Total Budget", yaxis_title="Incremental Outcome vs Current Allocation",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# Export
st.subheader("Export Recommendations")
export_df = alloc_df[["Channel", "Current", "Recommended", "Change ($)", "Change (%)"]].copy()
export_df.columns = ["Channel", "Current_Spend", "Recommended_Spend", "Change_Dollars", "Change_Percent"]

csv = export_df.to_csv(index=False)
st.download_button(
    "Download allocation as CSV", csv, "mmm_budget_allocation.csv", "text/csv",
    use_container_width=True
)

st.caption("""
**Important:** This optimization uses the fitted model's response curves with their estimated uncertainty.
Recommendations should be validated through controlled experiments (geo tests, holdouts) before implementation.
The model assumes channel independence and steady-state conditions; real-world dynamics may differ.
""")