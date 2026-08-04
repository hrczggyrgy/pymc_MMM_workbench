import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import arviz as az

from utils.data import ensure_demo_data, configured_data, validate_data
from utils import _lazy_import_modeling, _lazy_import_plotting

st.set_page_config(page_title="Model | MMM Workbench", layout="wide")
ensure_demo_data()

# Lazy imports
fit_bayesian_mmm, fit_bayesian_mmm_cached, *_ = _lazy_import_modeling()
line_with_band, *_ = _lazy_import_plotting()

df = configured_data()
channels = st.session_state.channel_cols
controls = st.session_state.control_cols

st.title("Bayesian MMM")
st.caption("Fit a Bayesian regression with adstock and saturation transformations. All parameters have posterior uncertainty.")

with st.expander("📖 How the model works", expanded=False):
    st.markdown("""
    **Model Specification:**
    ```
    sales_t ~ Normal(μ_t, σ)
    μ_t = α + Σ β_c × f_c(spend_c,t) + γ × controls_t + δ × trend_t + seasonality_t
    ```
    
    **Transformations (fixed, not estimated):**
    - **Adstock**: `f_c(spend) = Σ_{k=0}^{L} λ_c^k × spend_{t-k}` (geometric carryover, per-channel λ)
    - **Saturation**: `Hill(x) = x^α_c / (x^α_c + midpoint_c^α_c)` (diminishing returns, per-channel α)
    
    **Priors:**
    - α (intercept) ~ Normal(0, 1.5) on standardized scale
    - β (channel coefficients) ~ Normal(0, prior_strength) on standardized scale
    - σ (noise) ~ HalfNormal(1)
    
    **Why Bayesian?**
    - Full posterior distributions → credible intervals for all predictions
    - Regularization via priors prevents overfitting on short time series
    - Natural uncertainty propagation to scenarios and optimization
    
    **Diagnostics to check:**
    - **R-hat** ≈ 1.0 (chains converged)
    - **ESS** > 100 (effective sample size)
    - **Divergences** = 0 (no sampling pathologies)
    """)

if len(channels) < 2:
    st.warning("Configure at least two channels on the **Data** page.")
    st.stop()

# Model configuration
st.subheader("Model Configuration")

# Per-channel parameters with shared defaults
st.markdown("**Per-channel adstock decay (λ) & saturation strength (α):**")
st.caption("Each channel can have its own carryover and saturation profile. Search/TV typically have slower decay; Social/Display saturate faster.")

# Initialize session state for per-channel params
if "per_channel_decay" not in st.session_state:
    st.session_state.per_channel_decay = {c: 0.5 for c in channels}
if "per_channel_strength" not in st.session_state:
    st.session_state.per_channel_strength = {c: 1.5 for c in channels}

# Shared defaults for quick setup
c1, c2 = st.columns(2)
shared_decay = c1.slider("Default adstock decay (λ)", 0.0, 0.95, 0.5, 0.01, key="model_shared_decay",
                         help="Applied to all channels when you click 'Apply to all'")
shared_strength = c2.slider("Default saturation strength (α)", 0.3, 3.0, 1.5, 0.05, key="model_shared_strength",
                            help="Applied to all channels when you click 'Apply to all'")

if st.button("Apply defaults to all channels", key="apply_defaults"):
    for c in channels:
        st.session_state.per_channel_decay[c] = shared_decay
        st.session_state.per_channel_strength[c] = shared_strength
    st.rerun()

# Per-channel sliders
decay = {}
strength = {}
cols = st.columns(min(4, len(channels)))
for i, channel in enumerate(channels):
    with cols[i % len(cols)]:
        st.markdown(f"**{channel.title()}**")
        decay[channel] = st.slider(
            f"Decay λ_{channel}",
            0.0, 0.95,
            st.session_state.per_channel_decay.get(channel, 0.5),
            0.01,
            key=f"decay_{channel}",
            help=f"Carryover rate for {channel}. Higher = longer memory."
        )
        strength[channel] = st.slider(
            f"Strength α_{channel}",
            0.3, 3.0,
            st.session_state.per_channel_strength.get(channel, 1.5),
            0.05,
            key=f"strength_{channel}",
            help=f"Saturation steepness for {channel}. Higher = sharper diminishing returns."
        )
        st.session_state.per_channel_decay[channel] = decay[channel]
        st.session_state.per_channel_strength[channel] = strength[channel]

# Shared settings
c1, c2, c3, c4 = st.columns(4)
prior_scale = c1.select_slider("Prior strength for β", options=[0.5, 1.0, 1.5, 2.0], value=1.0, key="model_prior",
                               help="Lower = stronger regularization (shrinkage toward zero)")
seasonality = c2.toggle("Annual seasonality (sin/cos)", value=True, key="model_seasonality")
l_max = c3.slider("Carryover window (L)", 1, 20, 8, key="model_lmax",
                  help="Maximum lag periods for adstock sum")
quick = c4.toggle("Fast demo fit (250 draws)", value=True, key="model_quick",
                  help="250 tune + 250 draws per chain. Uncheck for production quality (700+ draws).")

midpoint_scale = st.select_slider("Midpoint scale", options=[0.5, 0.75, 1.0, 1.5, 2.0], value=1.0, key="model_midpoint",
                                   help="Scales channel midpoint relative to median spend")

# Out-of-sample validation
test_size = st.slider("Holdout fraction (out-of-sample)", 0.0, 0.3, 0.0, 0.05, key="model_test_size",
                      help="Fraction of data to hold out for validation. 0.2 = last 20% for testing.")

draws = 250 if quick else 700
tune = 250 if quick else 700

# Validation warnings
issues = validate_data(df, st.session_state.date_col, st.session_state.target_col, channels)
for level, msg in issues:
    getattr(st, level)(msg)

if st.button("Fit Bayesian MMM", type="primary", use_container_width=True):
    with st.spinner("Sampling posterior distributions... (this may take 30-120 seconds)"):
        try:
            # Convert per-channel dicts to tuples for cache key
            channels_tuple = tuple(channels)
            controls_tuple = tuple(controls)
            decay_tuple = tuple(decay[c] for c in channels)
            strength_tuple = tuple(strength[c] for c in channels)
            
            result = fit_bayesian_mmm_cached(
                df, st.session_state.date_col, st.session_state.target_col,
                channels_tuple, controls_tuple, decay_tuple, l_max, strength_tuple,
                midpoint_scale, seasonality, prior_scale, draws, tune, test_size=test_size
            )
            st.session_state.model_result = result
            st.success("Model fit complete!")
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")
            st.exception(exc)

result = st.session_state.get("model_result")
if not result:
    st.info("Configure model settings above and click **Fit Bayesian MMM** to see results.")
    st.stop()

# --- 1. Observed vs Predicted ---
st.subheader("1. Model Fit: Observed vs Posterior Prediction")
pred = result["prediction"]
st.plotly_chart(line_with_band(pred, "date", "mean", "low", "high",
                                "Observed vs Posterior Prediction", "observed"),
                use_container_width=True)

# Fit metrics
col1, col2, col3, col4 = st.columns(4)
rmse = np.sqrt(((pred.observed - pred["mean"]) ** 2).mean())
mape = np.mean(np.abs((pred.observed - pred["mean"]) / pred.observed)) * 100
r2 = 1 - ((pred.observed - pred["mean"]) ** 2).sum() / ((pred.observed - pred.observed.mean()) ** 2).sum()
col1.metric("RMSE (train)", f"{rmse:,.0f}")
col2.metric("MAPE (train)", f"{mape:.1f}%")
col3.metric("R² (train)", f"{r2:.3f}")
col4.metric("Periods", f"{len(pred)}")

# Out-of-sample validation results
if "test_prediction" in result and result["test_prediction"] is not None:
    st.markdown("---")
    st.subheader("🔍 Out-of-Sample Validation")
    test_pred = result["test_prediction"]
    test_metrics = result["test_metrics"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (test)", f"{test_metrics['rmse']:,.0f}", delta=f"{test_metrics['rmse'] - rmse:,.0f} vs train", delta_color="inverse")
    c2.metric("MAPE (test)", f"{test_metrics['mape']:.1f}%", delta=f"{test_metrics['mape'] - mape:.1f}% vs train", delta_color="inverse")
    c3.metric("R² (test)", f"{test_metrics['r2']:.3f}", delta=f"{test_metrics['r2'] - r2:.3f} vs train", delta_color="normal")
    c4.metric("Test periods", f"{test_metrics['n_test']}")
    
    # Test prediction chart
    st.plotly_chart(line_with_band(test_pred, "date", "mean", "low", "high",
                                    "Out-of-Sample: Observed vs Posterior Prediction", "observed"),
                    use_container_width=True)

# --- 2. Posterior Summary ---
st.subheader("2. Posterior Summary")
summary = result["summary"]

# Filter for key parameters
beta_rows = summary[summary["parameter"].str.startswith("beta")].copy()
beta_rows["channel"] = beta_rows["parameter"].str.extract(r"beta\[(.*?)\]")

# Add channel mapping
channel_map = {c: c for c in channels}
for idx, row in beta_rows.iterrows():
    feat = row["channel"]
    if feat in result["features"]:
        beta_rows.at[idx, "feature"] = feat

summary_fmt = summary.copy()
for col in ["mean", "sd", "hdi_3%", "hdi_97%"]:
    if col in summary_fmt.columns:
        summary_fmt[col] = summary_fmt[col].apply(lambda x: f"{x:.3f}")
for col in ["mcse_mean", "mcse_sd"]:
    if col in summary_fmt.columns:
        summary_fmt[col] = summary_fmt[col].apply(lambda x: f"{x:.4f}")
for col in ["ess_bulk", "ess_tail"]:
    if col in summary_fmt.columns:
        summary_fmt[col] = summary_fmt[col].apply(lambda x: f"{x:.0f}")
if "r_hat" in summary_fmt.columns:
    summary_fmt["r_hat"] = summary_fmt["r_hat"].apply(lambda x: f"{x:.3f}")

st.dataframe(
    summary_fmt,
    use_container_width=True, hide_index=True
)

# --- 3. Channel Coefficient Posteriors ---
st.subheader("3. Channel Effect Posteriors (β coefficients)")
beta_draws = result["beta_draws"]
feature_names = result["features"]

fig_beta = make_subplots(
    rows=1, cols=len(channels),
    subplot_titles=[c.title() for c in channels],
    horizontal_spacing=0.05
)

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

for i, ch in enumerate(channels):
    if ch in feature_names:
        idx = feature_names.index(ch)
        draws_ch = beta_draws[:, idx]
        
        # Density
        hist, bin_edges = np.histogram(draws_ch, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig_beta.add_trace(go.Scatter(
            x=bin_centers, y=hist,
            fill="tozeroy", fillcolor=f"rgba{tuple(list(bytes.fromhex(colors[i][1:])) + [0.3])}",
            line=dict(color=colors[i], width=2), name=ch, showlegend=False,
            hovertemplate="β: %{x:.3f}<br>Density: %{y:.2f}<extra></extra>"
        ), row=1, col=i+1)
        
        # Zero line
        fig_beta.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=i+1)
        
        # Mean marker
        mean_val = draws_ch.mean()
        fig_beta.add_vline(x=mean_val, line_dash="dash", line_color=colors[i],
                           annotation_text=f"{mean_val:.2f}", annotation_position="top", row=1, col=i+1)

fig_beta.update_layout(
    template="plotly_white",
    title="Posterior Distributions of Channel Coefficients (standardized)",
    margin=dict(l=10, r=10, t=60, b=10),
    height=350,
)
st.plotly_chart(fig_beta, use_container_width=True)

# Coefficient table with credible intervals
coeff_data = []
for ch in channels:
    if ch in feature_names:
        idx = feature_names.index(ch)
        draws_ch = beta_draws[:, idx]
        coeff_data.append({
            "Channel": ch,
            "Mean": f"{draws_ch.mean():.3f}",
            "Median": f"{np.median(draws_ch):.3f}",
            "SD": f"{draws_ch.std():.3f}",
            "5%": f"{np.quantile(draws_ch, 0.05):.3f}",
            "95%": f"{np.quantile(draws_ch, 0.95):.3f}",
            "P(β>0)": f"{(draws_ch > 0).mean():.1%}",
        })

coeff_df = pd.DataFrame(coeff_data)
st.dataframe(coeff_df, use_container_width=True, hide_index=True)

# --- 4. Channel Contribution Breakdown ---
st.subheader("4. Channel Contribution to Sales (Posterior Mean)")
contrib = result["contrib"]
total_contrib = contrib.sum().sort_values(ascending=False)

fig_contrib = go.Figure()
fig_contrib.add_trace(go.Bar(
    x=total_contrib.index, y=total_contrib.values,
    marker_color=[colors[i % len(colors)] for i in range(len(total_contrib))],
    text=[f"{v:,.0f}" for v in total_contrib.values],
    textposition="outside",
    hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
))
fig_contrib.update_layout(
    template="plotly_white",
    title="Total Modeled Contribution by Channel",
    yaxis_title="Contribution to sales (posterior mean)",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_contrib, use_container_width=True)

# Contribution over time
st.markdown("**Contribution Over Time**")
contrib_over_time = contrib.copy()
contrib_over_time["date"] = pred["date"]
fig_contrib_ts = go.Figure()
for i, ch in enumerate(channels):
    fig_contrib_ts.add_trace(go.Scatter(
        x=contrib_over_time["date"], y=contrib_over_time[ch],
        name=ch.title(), line=dict(color=colors[i % len(colors)], width=2),
        stackgroup="one", hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"
    ))
fig_contrib_ts.update_layout(
    template="plotly_white",
    title="Stacked Channel Contributions Over Time",
    yaxis_title="Contribution to sales",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_contrib_ts, use_container_width=True)

# --- 5. Model Diagnostics ---
st.subheader("5. Sampling Diagnostics")
idata = result["idata"]

# R-hat and ESS summary
diag_summary = az.summary(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.9)
max_rhat = diag_summary["r_hat"].max()
min_ess_bulk = diag_summary["ess_bulk"].min()
min_ess_tail = diag_summary["ess_tail"].min()

c1, c2, c3 = st.columns(3)
c1.metric("Max R-hat", f"{max_rhat:.3f}", delta="OK" if max_rhat < 1.01 else "⚠️ High", delta_color="normal" if max_rhat < 1.01 else "inverse")
c2.metric("Min ESS (bulk)", f"{min_ess_bulk:.0f}", delta="OK" if min_ess_bulk > 100 else "⚠️ Low", delta_color="normal" if min_ess_bulk > 100 else "inverse")
c3.metric("Min ESS (tail)", f"{min_ess_tail:.0f}", delta="OK" if min_ess_tail > 100 else "⚠️ Low", delta_color="normal" if min_ess_tail > 100 else "inverse")

# Trace plots for key parameters
st.markdown("**Trace Plots (first 4 chains combined)**")
trace_vars = ["alpha", "sigma"] + [f"beta[{ch}]" for ch in channels if f"beta[{ch}]" in idata.posterior.data_vars]

fig_trace = make_subplots(
    rows=len(trace_vars), cols=1,
    subplot_titles=trace_vars,
    vertical_spacing=0.03,
    shared_xaxes=True
)

for i, var in enumerate(trace_vars):
    if var in idata.posterior.data_vars:
        samples = idata.posterior[var].values.flatten()
        # Subsample for plotting
        step = max(1, len(samples) // 2000)
        samples_plot = samples[::step]
        fig_trace.add_trace(go.Scatter(
            y=samples_plot, mode="lines", line=dict(color=colors[i % len(colors)], width=0.5),
            showlegend=False, hovertemplate="Sample: %{y:.3f}<extra></extra>"
        ), row=i+1, col=1)

fig_trace.update_layout(
    template="plotly_white",
    height=200 * len(trace_vars),
    margin=dict(l=10, r=10, t=50, b=10),
    showlegend=False,
)
st.plotly_chart(fig_trace, use_container_width=True)

# Pair plot for beta coefficients (if few channels)
if len(channels) <= 4:
    st.markdown("**Posterior Pair Plot (Channel Coefficients)**")
    try:
        beta_data = idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "feature").values
        beta_df = pd.DataFrame(beta_data, columns=feature_names)
        # Only keep channel columns
        beta_df = beta_df[[c for c in channels if c in beta_df.columns]]
        
        import plotly.express as px
        fig_pair = px.scatter_matrix(
            beta_df, dimensions=beta_df.columns,
            opacity=0.3, title="Posterior Correlations Between Channel Effects"
        )
        fig_pair.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_pair, use_container_width=True)
    except Exception:
        pass

# --- 6. Residuals ---
st.subheader("6. Residual Diagnostics")
residuals = pred.observed - pred["mean"]

c1, c2 = st.columns(2)
with c1:
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=pred["date"], y=residuals,
        mode="markers", marker=dict(color="#4C72B0", size=6, opacity=0.6),
        name="Residuals"
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_resid.update_layout(template="plotly_white", title="Residuals Over Time",
                            yaxis_title="Observed - Predicted", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_resid, use_container_width=True)

with c2:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals, nbinsx=30, marker_color="#4C72B0", opacity=0.7,
        name="Residuals"
    ))
    # Normal overlay
    from scipy import stats
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max() - residuals.min()) / 30
    fig_hist.add_trace(go.Scatter(x=x_norm, y=y_norm, line=dict(color="#EF4444", width=2), name="Normal fit"))
    fig_hist.update_layout(template="plotly_white", title="Residual Distribution",
                           xaxis_title="Residual", yaxis_title="Count", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# Save model for other pages
st.caption("Model saved. Navigate to **Scenarios** or **Optimization** to use the fitted model.")