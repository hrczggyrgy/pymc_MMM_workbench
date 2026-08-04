import plotly.express as px
import streamlit as st
from utils.data import ensure_demo_data,configured_data,validate_data
from utils.modeling import fit_bayesian_mmm
from utils.plotting import line_with_band
st.set_page_config(page_title="Model | MMM Workbench",page_icon="🧠",layout="wide"); ensure_demo_data(); df=configured_data(); channels=st.session_state.channel_cols; controls=st.session_state.control_cols; st.title("Bayesian MMM"); st.info("Fits a real PyMC Bayesian regression over adstocked and saturated media features.")
if len(channels)<2: st.warning("Configure at least two channels on Data."); st.stop()
a,b,c,d=st.columns(4); decay=a.slider("Shared adstock decay",0.,.95,.5,.05); strength=b.slider("Shared saturation strength",.3,3.,1.5,.1); prior=c.select_slider("Prior strength",options=[.5,1.,1.5,2.],value=1.); seasonality=d.toggle("Annual seasonality",value=True); l_max=st.slider("Carryover window",1,20,8); quick=st.toggle("Fast demo fit (250 draws)",value=True)
if st.button("Fit Bayesian MMM",type="primary"):
    with st.spinner("Sampling posterior distributions..."):
        try: st.session_state.model_result=fit_bayesian_mmm(df,st.session_state.date_col,st.session_state.target_col,channels,controls,decay,l_max,strength,1.,seasonality,prior,250 if quick else 700,250 if quick else 700); st.success("Model fit complete.")
        except Exception as exc: st.error(f"Model fitting failed: {exc}")
result=st.session_state.get("model_result")
if not result: st.caption("Fit a model to view posterior results."); st.stop()
pred=result["prediction"]; st.plotly_chart(line_with_band(pred,"date","mean","low","high","Observed vs posterior prediction","observed"),use_container_width=True); st.markdown("### Posterior summary"); st.dataframe(result["summary"],use_container_width=True)
contrib=result["contrib"].sum().sort_values(ascending=False).reset_index(); contrib.columns=["Channel","Estimated contribution"]; st.plotly_chart(px.bar(contrib,x="Channel",y="Estimated contribution",color="Estimated contribution",color_continuous_scale="Teal",template="plotly_white"),use_container_width=True)
rmse=((pred.observed-pred["mean"])**2).mean()**.5; st.caption(f"Diagnostic snapshot: RMSE {rmse:,.0f}. Check R-hat near 1 before decision use.")
