import streamlit as st
from utils.data import ensure_demo_data, set_data
from utils.simulation import generate_demo_data

st.set_page_config(page_title="Home | MMM Workbench", page_icon=":material/home:", layout="wide")
ensure_demo_data()

st.title("Marketing Mix Modeling Workbench")
st.subheader("Turn marketing spend into transparent, uncertainty-aware decisions.")

# Demo data button
c1, c2 = st.columns([1, 3])
with c1:
    if st.button(":material/rocket_launch: Try Demo Data", type="primary", use_container_width=True):
        set_data(generate_demo_data(), "Built-in synthetic demo")
        st.success("Demo data loaded! Navigate to **Data** to review, then **Effect Explorer**.")
with c2:
    st.info("First time? Click **Try Demo Data** to see the full workflow with synthetic data.")

st.markdown("---")

# Workflow guide
st.markdown("### :material/checklist: Your Guided Workflow")
steps = [
    ("1. Data", ":material/folder:", "Upload a CSV or use demo data. Validate schema, select date/target/channels/controls."),
    ("2. Explore", ":material/science:", "Learn carryover (adstock) and diminishing returns (saturation) interactively per channel."),
    ("3. Fit", ":material/psychology:", "Estimate media effects with a real Bayesian model (PyMC). Get full posterior uncertainty."),
    ("4. Simulate", ":material/target:", "Test spend plans: 'What if Search +20%?' See credible intervals, not point estimates."),
    ("5. Optimize", ":material/tune:", "Allocate a fixed budget across channels to maximize expected outcome under constraints."),
]

cols = st.columns(5)
for col, (title, icon, desc) in zip(cols, steps):
    with col:
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

st.markdown("---")

# What is MMM
with st.expander(":material/book: What is Marketing Mix Modeling?", expanded=True):
    st.markdown("""
    **Marketing Mix Modeling (MMM)** estimates how marketing activities (and other factors) drive a business outcome over time.
    
    **Typical questions MMM answers:**
    - How much did each channel contribute to sales?
    - What's the ROI/ROAS of each channel?
    - How should I allocate my budget next quarter?
    - What happens if I increase/decrease spend on a channel?
    
    **Why Bayesian MMM?**
    - **Uncertainty quantification**: Credible intervals on every estimate, not false precision
    - **Regularization**: Priors prevent overfitting on short time series (52-104 weeks typical)
    - **Decision support**: Full posterior → propagate uncertainty to scenarios & optimization
    - **Interpretability**: Posterior distributions show *plausible* effect sizes, not just point estimates
    
    **Core concepts in this workbench:**
    - **Adstock (carryover)**: Advertising effects persist over time (geometric decay)
    - **Saturation (diminishing returns)**: Each additional dollar is less effective (Hill function)
    - **Controls**: Price, promotions, holidays, trend, seasonality
    - **Posterior**: Plausible parameter values after combining data + priors
    """)

# Glossary
with st.expander(":material/menu_book: Glossary", expanded=False):
    st.markdown("""
    | Term | Definition |
    |------|------------|
    | **Adstock** | Advertising carryover: spend today affects future periods (geometric decay λ) |
    | **Saturation** | Diminishing returns: response curve flattens at high spend (Hill function) |
    | **Posterior** | Distribution of plausible parameter values given data + priors |
    | **Credible Interval** | Bayesian uncertainty range (e.g., 90% CI = 5th–95th percentile) |
    | **R-hat** | Convergence diagnostic (≈1.0 means chains converged) |
    | **ESS** | Effective Sample Size (higher = more independent draws) |
    | **ROAS** | Return on Ad Spend: incremental revenue / spend |
    | **Marginal ROAS** | Derivative of response curve: return on the *next* dollar |
    | **Prior** | Assumptions about parameters before seeing data (regularization) |
    | **Lift** | Incremental outcome from a scenario vs. baseline |
    """)

# Caveats
st.warning("""
**Important Caveats**
- This is an **analytical prototype**, not causal proof. MMM identifies associations, not causation.
- **Validate before deciding**: Check holdout performance, run calibration experiments (geo tests), review with domain experts.
- **Data quality matters**: Missing values, incorrect spend definitions, aggregation level all affect results.
- **Model assumptions**: Shared adstock/saturation across channels, no interactions, steady-state optimization.
- **Extrapolation risk**: Response curves beyond observed spend range are highly uncertain.
""")

# Quick start
st.markdown("### :material/play_arrow: Quick Start")
st.markdown("""
1. **Click "Try Demo Data"** above (or upload your CSV on the **Data** page)
2. Go to **Data** → verify column mapping → **Save configuration**
3. **Effect Explorer** → play with adstock/saturation sliders per channel
4. **Model** → adjust priors/seasonality → **Fit Bayesian MMM**
5. **Scenarios** → test "what-if" spend changes
6. **Optimization** → set budget & constraints → find optimal allocation
""")