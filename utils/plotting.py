import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def line_with_band(df,x,mean,low,high,title,observed=None):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df[x],y=df[high],mode="lines",line=dict(width=0),showlegend=False)); fig.add_trace(go.Scatter(x=df[x],y=df[low],mode="lines",fill="tonexty",fillcolor="rgba(76,114,176,.18)",line=dict(width=0),name="90% credible interval")); fig.add_trace(go.Scatter(x=df[x],y=df[mean],mode="lines",line=dict(color="#4C72B0",width=3),name="Posterior mean"))
    if observed: fig.add_trace(go.Scatter(x=df[x],y=df[observed],mode="markers",marker=dict(color="#1F2937",size=5),name="Observed"))
    fig.update_layout(title=title,template="plotly_white",legend_orientation="h",margin=dict(l=10,r=10,t=50,b=10)); return fig
def allocation_chart(current,allocation):
    df=pd.DataFrame({"channel":list(allocation),"Current":[current[c] for c in allocation],"Recommended":list(allocation.values())}).melt("channel",var_name="Plan",value_name="Spend")
    return px.bar(df,x="channel",y="Spend",color="Plan",barmode="group",template="plotly_white",title="Current vs recommended average-period allocation",color_discrete_sequence=["#94A3B8","#0F766E"])
