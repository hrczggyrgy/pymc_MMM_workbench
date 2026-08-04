import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data import ensure_demo_data, set_data, detect_schema, validate_and_clean_upload, validate_data
from utils.simulation import generate_demo_data

st.set_page_config(page_title="Data | MMM Workbench", page_icon=":material/folder:", layout="wide")
ensure_demo_data()

st.title("Data Setup")
st.caption("Load a time-series CSV or use demo data. Configure column roles and validate.")

with st.expander("Data Requirements", expanded=False):
    st.markdown("""
    **Expected CSV format:**
    - **Date column**: Parseable dates (weekly or daily), one row per period
    - **Target column**: Numeric outcome (sales, revenue, conversions, orders)
    - **Channel columns**: 2+ numeric media spend columns (search, social, video, display, tv, radio, etc.)
    - **Control columns** (optional): Price, promotions, holidays, trend, seasonality indicators
    
    **Data quality checks (automatic):**
    - Missing values → warning
    - Duplicate dates → warning
    - Negative spend → warning
    - Short time series (<30 periods) → warning
    - Non-monotonic dates → info (auto-sorted)
    
    **Tips:**
    - Use consistent time granularity (weekly recommended)
    - Ensure spend is in same currency/units across channels
    - Include at least 52 weeks (1 year) for seasonality estimation
    - Controls help isolate media effects from confounders
    """)

# File upload
uploaded = st.file_uploader("Upload CSV", type="csv", help="CSV with date, target, channel spends, and optional controls")

if uploaded:
    try:
        df_uploaded = pd.read_csv(uploaded)
        set_data(df_uploaded, uploaded.name)
        st.toast(f"Loaded {uploaded.name} ({len(df_uploaded):,} rows, {len(df_uploaded.columns)} columns)")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

if st.button("Reset to Demo Data", use_container_width=True):
    set_data(generate_demo_data(), "Built-in synthetic demo")
    st.rerun()

# Current data preview
df = st.session_state.data
st.markdown("---")
st.subheader("Data Preview")
c1, c2, c3 = st.columns(3)
c1.metric("Source", st.session_state.source)
c2.metric("Rows", f"{len(df):,}")
c3.metric("Columns", f"{len(df.columns)}")

st.dataframe(df.head(15), use_container_width=True)

# Schema detection & configuration
st.markdown("---")
st.subheader("Column Configuration")

columns = list(df.columns)
auto_date, auto_target, auto_channels, auto_controls = detect_schema(df)

c1, c2 = st.columns(2)
date_col = c1.selectbox(
    "Date column",
    columns,
    index=columns.index(st.session_state.date_col) if st.session_state.date_col in columns else 0,
    help="Time index (will be parsed as datetime)"
)
target_col = c2.selectbox(
    "Target / outcome column",
    columns,
    index=columns.index(st.session_state.target_col) if st.session_state.target_col in columns else 0,
    help="Business outcome to model (sales, revenue, conversions)"
)

numeric = df.select_dtypes(include="number").columns.tolist()
channel_options = [c for c in numeric if c != target_col]

channels = st.multiselect(
    "Media-spend columns (2+ required)",
    channel_options,
    default=[c for c in st.session_state.channel_cols if c in channel_options],
    help="Marketing channels with spend data. At least 2 required."
)

control_options = [c for c in numeric if c not in channels and c != target_col]
controls = st.multiselect(
    "Optional control columns",
    control_options,
    default=[c for c in st.session_state.control_cols if c in control_options],
    help="Confounders: price, promotions, holidays, etc."
)

# Save configuration & run full validation
if st.button("Save Data Configuration", type="primary", use_container_width=True):
    st.session_state.date_col = date_col
    st.session_state.target_col = target_col
    st.session_state.channel_cols = channels
    st.session_state.control_cols = controls
    st.session_state.pop("model_result", None)
    
    # Run comprehensive validation
    with st.spinner("Validating and cleaning data..."):
        result = validate_and_clean_upload(
            st.session_state.data, date_col, target_col, channels, controls
        )
    
    st.session_state.validation_result = result
    
    if result["usable"]:
        st.toast("Configuration saved! Proceed to Effect Explorer or Model.")
    else:
        st.error("Configuration has errors that must be fixed before proceeding.")

# Validation results
st.markdown("---")
st.subheader("Data Validation")

if "validation_result" in st.session_state:
    result = st.session_state.validation_result
    
    # Errors (blocking)
    if result["errors"]:
        st.error("**Blocking Errors:**")
        for err in result["errors"]:
            st.error(f"- {err}")
    
    # Warnings (non-blocking)
    if result["warnings"]:
        with st.expander("Warnings", expanded=True):
            for warn in result["warnings"]:
                st.warning(f"- {warn}")
    
    # Channel info table
    if result.get("channel_info"):
        st.subheader("Channel Diagnostics")
        info_rows = []
        for col, info in result["channel_info"].items():
            status = "Excluded" if col in result["excluded_channels"] else "Valid"
            info_rows.append({
                "Channel": col,
                "Status": status,
                "Missing": f"{info['n_missing']}/{info['n_total']}",
                "Zeros": info['n_zero'],
                "Negatives": info['n_negative'],
                "Mean": f"{info['mean']:,.0f}",
                "Median": f"{info['median']:,.0f}",
                "Std": f"{info['std']:,.0f}",
                "Min": f"{info['min']:,.0f}",
                "Max": f"{info['max']:,.0f}",
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)
    
    # Excluded channels
    if result["excluded_channels"]:
        st.info(f"Excluded channels: {', '.join(result['excluded_channels'])}")
else:
    # Legacy validation
    issues = validate_data(df, date_col, target_col, channels)
    if not issues:
        st.success("No issues detected")
    else:
        for level, message in issues:
            getattr(st, level)(message)

# Quick visualizations
if len(channels) >= 1:
    st.markdown("---")
    st.subheader("Quick Visualizations")
    
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df_viz = df.copy()
        df_viz["_date_parsed"] = dates
        df_viz = df_viz.sort_values("_date_parsed")
        
        fig = px.line(df_viz, x="_date_parsed", y=target_col, title=f"{target_col} Over Time")
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Channel spends
        if len(channels) > 0:
            fig2 = px.line(df_viz, x="_date_parsed", y=channels, title="Channel Spends Over Time")
            fig2.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation heatmap
            corr_cols = channels + ([target_col] if target_col in numeric else []) + controls
            if len(corr_cols) > 1:
                corr = df_viz[corr_cols].corr()
                fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                 title="Correlation Matrix", zmin=-1, zmax=1)
                fig3.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.info(f"Could not generate visualizations: {e}")

# Data summary
st.markdown("---")
st.subheader("Data Summary")
summary_data = []
for col in df.columns:
    dtype = str(df[col].dtype)
    missing = df[col].isna().sum()
    missing_pct = missing / len(df) * 100
    if pd.api.types.is_numeric_dtype(df[col]):
        summary_data.append({
            "Column": col, "Type": dtype, "Missing": f"{missing} ({missing_pct:.1f}%)",
            "Min": f"{df[col].min():,.2f}", "Max": f"{df[col].max():,.2f}",
            "Mean": f"{df[col].mean():,.2f}", "Std": f"{df[col].std():,.2f}"
        })
    else:
        summary_data.append({
            "Column": col, "Type": dtype, "Missing": f"{missing} ({missing_pct:.1f}%)",
            "Min": "—", "Max": "—", "Mean": "—", "Std": "—"
        })

st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)