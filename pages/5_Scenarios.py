import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.data import ensure_demo_data
from utils.modeling import scenario_lift
st.set_page_config(page_title="Scenarios | MMM Workbench",page_icon="🎯",layout="wide"); ensure_demo_data(); result=st.session_state.get("model_result"); st.title("Scenario Planner"); st.info("Test future average-period spend versus the fitted baseline, with posterior uncertainty.")
if not result: st.warning("Fit a model first."); st.stop()
plan={}; cols=st.columns(len(result["channels"]))
for col,channel in zip(cols,result["channels"]):
    baseline=result["current_spend"][channel]; change=col.slider(f"{channel.title()} change",-50,100,0,5,key=f"scenario_{channel}"); plan[channel]=baseline*(1+change/100); col.caption(f"Baseline {baseline:,.0f} → Plan {plan[channel]:,.0f}")
lift=scenario_lift(plan,result); mean,low,high=lift.mean(),np.quantile(lift,.05),np.quantile(lift,.95); a,b,c=st.columns(3); a.metric("Expected incremental outcome",f"{mean:,.0f}"); b.metric("90% credible interval",f"{low:,.0f} to {high:,.0f}"); c.metric("Plan budget change",f"{(sum(plan.values())/sum(result['current_spend'].values())-1)*100:+.1f}%")
compare=pd.DataFrame({"channel":result["channels"],"Baseline":[result["current_spend"][c] for c in result["channels"]],"Scenario":[plan[c] for c in result["channels"]]}).melt("channel",var_name="Plan",value_name="Spend"); st.bar_chart(compare,x="channel",y="Spend",color="Plan")
if low<0<high: st.warning("The interval spans zero: modeled impact is uncertain.")
else: st.success("The modeled interval is directionally consistent; validate with experiments where possible.")
