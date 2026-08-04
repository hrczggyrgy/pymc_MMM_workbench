import pandas as pd
import streamlit as st
from utils.data import ensure_demo_data
from utils.modeling import optimize_budget
from utils.plotting import allocation_chart
st.set_page_config(page_title="Optimization | MMM Workbench",page_icon="⚙️",layout="wide"); ensure_demo_data(); result=st.session_state.get("model_result"); st.title("Budget Optimizer"); st.info("Allocate fixed average-period budget across channels using fitted response curves and practical constraints.")
if not result: st.warning("Fit a model first."); st.stop()
current_total=sum(result["current_spend"].values()); total=st.number_input("Total average-period budget",min_value=100.,value=float(current_total),step=1000.); minimums={}; maximums={}
for channel in result["channels"]:
    current=result["current_spend"][channel]; a,b,c=st.columns(3); a.write(f"**{channel.title()}** (current: {current:,.0f})"); minimums[channel]=b.number_input(f"Minimum {channel}",0.,value=float(current*.25),step=500.,key=f"min_{channel}"); maximums[channel]=c.number_input(f"Maximum {channel}",0.,value=float(current*3),step=500.,key=f"max_{channel}")
if st.button("Find recommended allocation",type="primary"):
    try: st.session_state.optimization=optimize_budget(total,minimums,maximums,result)
    except Exception as exc: st.error(str(exc))
if "optimization" in st.session_state:
    allocation,lift=st.session_state.optimization; a,b,c=st.columns(3); a.metric("Expected improvement vs current",f"{lift.mean():,.0f}"); b.metric("90% credible interval",f"{lift.min():,.0f} to {lift.max():,.0f}"); c.metric("Recommended budget",f"{sum(allocation.values()):,.0f}")
    table=pd.DataFrame({"Channel":list(allocation),"Current":[result["current_spend"][c] for c in allocation],"Recommended":list(allocation.values())}); table["Change"]=table.Recommended-table.Current; st.dataframe(table.style.format({"Current":"{:,.0f}","Recommended":"{:,.0f}","Change":"{:+,.0f}"}),use_container_width=True); st.plotly_chart(allocation_chart(result["current_spend"],allocation),use_container_width=True)
