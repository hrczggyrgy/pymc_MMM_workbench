import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.data import ensure_demo_data,configured_data
from utils.modeling import geometric_adstock,hill_saturation
st.set_page_config(page_title="Effect Explorer | MMM Workbench",page_icon="🔬",layout="wide"); ensure_demo_data(); df=configured_data(); channels=st.session_state.channel_cols; st.title("Effect Explorer"); st.info("**Adstock** spreads impact into later periods; **saturation** captures diminishing returns.")
if not channels: st.warning("Choose a channel on Data."); st.stop()
channel=st.selectbox("Channel",channels); a,b,c,d=st.columns(4); decay=a.slider("Adstock decay",0.,.95,.5,.05); window=b.slider("Carryover window",1,20,8); strength=c.slider("Saturation strength",.3,3.,1.5,.1); raw=df[channel].to_numpy(float); default_mid=float(np.median(raw[raw>0])) if np.any(raw>0) else 1; midpoint=d.slider("Saturation midpoint",1.,float(max(raw.max()*2,2)),default_mid)
adstock=geometric_adstock(raw,decay,window); x=np.linspace(0,max(adstock.max()*1.5,midpoint*2),200); y=hill_saturation(x,strength,midpoint)
fig=go.Figure(); fig.add_trace(go.Scatter(x=df[st.session_state.date_col],y=raw,name="Raw spend")); fig.add_trace(go.Scatter(x=df[st.session_state.date_col],y=adstock,name="Adstocked spend")); fig.update_layout(title="Spend with carryover",template="plotly_white"); st.plotly_chart(fig,use_container_width=True)
a,b=st.columns(2); curve=go.Figure(go.Scatter(x=x,y=y,fill="tozeroy")); curve.update_layout(title="Saturation response curve",template="plotly_white"); a.plotly_chart(curve,use_container_width=True); marg=go.Figure(go.Scatter(x=x,y=np.gradient(y,x))); marg.update_layout(title="Marginal response",template="plotly_white"); b.plotly_chart(marg,use_container_width=True)
st.markdown(f"**Interpretation:** decay {decay:.2f} spreads spend over roughly {window} future periods. Near midpoint {midpoint:,.0f}, the channel reaches half its modeled maximum response.")
