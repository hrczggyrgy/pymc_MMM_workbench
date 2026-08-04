import pandas as pd
import streamlit as st
from utils.data import ensure_demo_data,set_data,validate_data
from utils.modeling import generate_demo_data
st.set_page_config(page_title="Data | MMM Workbench",page_icon="🗂️",layout="wide"); ensure_demo_data(); st.title("Data setup"); st.caption("Load a time-series CSV or use demo data.")
uploaded=st.file_uploader("Upload CSV",type="csv")
if uploaded:
    try: set_data(pd.read_csv(uploaded),uploaded.name); st.success(f"Loaded {uploaded.name}")
    except Exception as exc: st.error(f"Could not read CSV: {exc}")
if st.button("Reset to demo data"): set_data(generate_demo_data(),"Built-in synthetic demo")
df=st.session_state.data; st.write(f"**Source:** {st.session_state.source} · **{len(df):,} rows** · **{len(df.columns)} columns**"); st.dataframe(df.head(12),use_container_width=True)
columns=list(df.columns); a,b=st.columns(2); date_col=a.selectbox("Date column",columns,index=columns.index(st.session_state.date_col) if st.session_state.date_col in columns else 0); target_col=b.selectbox("Target / outcome column",columns,index=columns.index(st.session_state.target_col) if st.session_state.target_col in columns else 0)
numeric=df.select_dtypes(include="number").columns.tolist(); channels=st.multiselect("Media-spend columns",numeric,default=[c for c in st.session_state.channel_cols if c in numeric]); controls=st.multiselect("Optional control columns",[c for c in numeric if c not in channels and c!=target_col],default=[c for c in st.session_state.control_cols if c in numeric and c not in channels])
if st.button("Save data configuration",type="primary"):
    st.session_state.date_col,date_col; st.session_state.date_col=date_col; st.session_state.target_col=target_col; st.session_state.channel_cols=channels; st.session_state.control_cols=controls; st.session_state.pop("model_result",None); st.success("Configuration saved.")
st.markdown("### Validation")
for level,message in validate_data(df,date_col,target_col,channels): getattr(st,level)(message)
