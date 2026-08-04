import streamlit as st
from utils.data import ensure_demo_data,set_data
from utils.modeling import generate_demo_data
st.set_page_config(page_title="Home | MMM Workbench",page_icon="🏠",layout="wide"); ensure_demo_data()
st.title("Marketing Mix Modeling Workbench"); st.subheader("Turn marketing spend into transparent, uncertainty-aware decisions.")
if st.button("Try demo data",type="primary"): set_data(generate_demo_data(),"Built-in synthetic demo"); st.success("Demo data is ready. Continue to Data, then Effect Explorer.")
st.markdown("### Your guided workflow")
for col,title,text in zip(st.columns(5),["1. Load","2. Explore","3. Fit","4. Simulate","5. Optimize"],["Upload a CSV or use the demo.","Learn carryover and diminishing returns.","Estimate media effects with PyMC.","Test spend plans and uncertainty.","Allocate a constrained budget."]): col.markdown(f"**{title}**\n\n{text}")
st.info("Marketing mix modeling estimates how media, promotions, price, trend, and seasonality relate to a business outcome over time. This workbench uses Bayesian estimates so recommendations include uncertainty rather than one deceptively certain number.")
with st.expander("Glossary",expanded=True): st.markdown("- **Adstock:** Advertising can influence later periods, not only the week it ran.\n- **Saturation:** Additional spend produces diminishing returns at high levels.\n- **Posterior:** Plausible parameter values after combining data and assumptions.\n- **ROAS:** Return on ad spend.\n- **Credible interval:** Bayesian uncertainty range.")
st.warning("Use this as an analytical prototype. MMM requires careful data collection, calibration, and expert review before high-stakes decisions.")
