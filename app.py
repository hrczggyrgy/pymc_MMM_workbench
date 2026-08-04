import streamlit as st
from utils.data import ensure_demo_data

st.set_page_config(page_title="PyMC MMM Workbench", page_icon=":material/analytics:", layout="wide")
ensure_demo_data()
st.logo("https://cdn.jsdelivr.net/npm/@mdi/svg@7.4.45/svg/chart-line.svg", size="large")
st.title("PyMC MMM Workbench")
st.subheader("A guided Bayesian marketing mix modeling lab")
st.markdown("Use the pages in the sidebar to load data, explore carryover and diminishing returns, fit a Bayesian model, test scenarios, and optimize your budget.")

left, right = st.columns(2)
with left:
    st.info("**Start here:** Open **Home** for a guided overview or **Data** to upload a CSV. Demo data is loaded automatically.")
with right:
    st.success("**Modeling note:** The Model page fits a real PyMC Bayesian regression over adstock and saturation transformed media features.")

st.caption("Decision support, not causal proof. Validate data quality, calibration assumptions, and business constraints before using recommendations operationally.")
