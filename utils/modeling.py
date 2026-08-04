from __future__ import annotations
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.optimize import minimize

def geometric_adstock(values: np.ndarray, decay: float = 0.5, l_max: int = 8) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.convolve(x, decay ** np.arange(l_max + 1), mode="full")[:len(x)]

def hill_saturation(values: np.ndarray, strength: float = 1.5, midpoint: float | None = None) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0)
    midpoint = float(np.median(x[x > 0])) if midpoint is None and np.any(x > 0) else (midpoint or 1.0)
    midpoint = max(float(midpoint), 1e-6)
    power = max(float(strength), 0.1)
    return x**power / (x**power + midpoint**power + 1e-9)

def transform_media(values, decay=.5, l_max=8, strength=1.5, midpoint=None):
    return hill_saturation(geometric_adstock(np.asarray(values), decay, l_max), strength, midpoint)

def generate_demo_data(n_periods: int = 104, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed); t = np.arange(n_periods)
    channels = {"search":np.clip(rng.gamma(5,1500,n_periods)*(1+.15*np.sin(t/8)),500,None),"social":np.clip(rng.gamma(4,1200,n_periods)*(1+.2*np.cos(t/10)),300,None),"video":np.clip(rng.gamma(3,2100,n_periods)*(1+.25*np.sin(t/15)),300,None),"display":np.clip(rng.gamma(4,700,n_periods),150,None)}
    price = 100+2*np.sin(t/12)+rng.normal(0,.8,n_periods); promo=rng.binomial(1,.14,n_periods)
    sales=52000+110*t+3500*np.sin(2*np.pi*t/52)-230*price+8500*promo
    for name,decay,strength,mid,beta in [("search",.58,1.35,23000,11000),("social",.42,1.7,18000,7000),("video",.72,1.2,32000,14000),("display",.3,1.9,10000,4000)]: sales += beta*transform_media(channels[name],decay,8,strength,mid)
    sales += rng.normal(0,2500,n_periods)
    return pd.DataFrame({"date":pd.date_range("2024-01-07",periods=n_periods,freq="W-SUN"),**channels,"price":price,"promo":promo,"sales":np.maximum(sales,1000)})

def prepare_features(df,date_col,target_col,channels,controls,decay,l_max,strength,midpoint_scale,seasonality):
    data=df.copy().sort_values(date_col).dropna(subset=[target_col]+channels); media={}; params={}
    for channel in channels:
        raw=data[channel].clip(lower=0).to_numpy(float); midpoint=max(np.median(raw[raw>0])*midpoint_scale if np.any(raw>0) else 1,1)
        media[channel]=transform_media(raw,decay,l_max,strength,midpoint); params[channel]={"decay":decay,"l_max":l_max,"strength":strength,"midpoint":midpoint}
    X=pd.DataFrame(media,index=data.index)
    for col in controls:
        if col in data: X[col]=pd.to_numeric(data[col],errors="coerce").fillna(data[col].median())
    X["trend"]=np.linspace(0,1,len(data))
    if seasonality:
        week=pd.to_datetime(data[date_col]).dt.isocalendar().week.astype(float).to_numpy(); X["sin_annual"]=np.sin(2*np.pi*week/52); X["cos_annual"]=np.cos(2*np.pi*week/52)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0); means,stds=X.mean(),X.std().replace(0,1)
    return data,X,(X-means)/stds,pd.to_numeric(data[target_col],errors="coerce").to_numpy(float),means,stds,params

def fit_bayesian_mmm(df,date_col,target_col,channels,controls,decay=.5,l_max=8,strength=1.5,midpoint_scale=1.,seasonality=True,prior_scale=1.,draws=500,tune=500,seed=42):
    data,X,X_scaled,y,means,stds,params=prepare_features(df,date_col,target_col,channels,controls,decay,l_max,strength,midpoint_scale,seasonality); y_mean,y_std=y.mean(),max(y.std(),1); y_scaled=(y-y_mean)/y_std
    coords={"feature":X.columns.tolist(),"obs":np.arange(len(data))}
    with pm.Model(coords=coords):
        x=pm.Data("x",X_scaled.to_numpy(),dims=("obs","feature")); alpha=pm.Normal("alpha",0,1.5); beta=pm.Normal("beta",0,prior_scale,dims="feature"); sigma=pm.HalfNormal("sigma",1); mu=pm.Deterministic("mu",alpha+pm.math.dot(x,beta),dims="obs")
        pm.Normal("likelihood",mu=mu,sigma=sigma,observed=y_scaled,dims="obs"); idata=pm.sample(draws=draws,tune=tune,chains=2,cores=1,target_accept=.9,progressbar=False,random_seed=seed,return_inferencedata=True); posterior_mu=idata.posterior["mu"].stack(sample=("chain","draw")).values
    pred_samples=posterior_mu*y_std+y_mean; prediction=pd.DataFrame({"date":data[date_col],"observed":y,"mean":pred_samples.mean(axis=1),"low":np.quantile(pred_samples,.05,axis=1),"high":np.quantile(pred_samples,.95,axis=1)})
    beta_draws=idata.posterior["beta"].stack(sample=("chain","draw")).transpose("sample","feature").values; feature_names=X.columns.tolist(); summary=az.summary(idata,var_names=["alpha","beta","sigma"],hdi_prob=.9).reset_index().rename(columns={"index":"parameter"})
    contrib=pd.DataFrame({c:beta_draws[:,feature_names.index(c)].mean()*X_scaled[c].to_numpy()*y_std for c in channels})
    return {"idata":idata,"prediction":prediction,"summary":summary,"contrib":contrib,"features":feature_names,"beta_draws":beta_draws,"means":means,"stds":stds,"y_mean":y_mean,"y_std":y_std,"params":params,"channels":channels,"current_spend":{c:float(data[c].mean()) for c in channels}}

def response_samples(spend,channel,result,n=600):
    p=result["params"][channel]; transformed=hill_saturation(np.array([spend/(1-p["decay"]+1e-6)]),p["strength"],p["midpoint"])[0]; scaled=(transformed-result["means"][channel])/result["stds"][channel]; beta=result["beta_draws"][:,result["features"].index(channel)]
    return beta[:n]*scaled*result["y_std"]
def scenario_lift(plan,result):
    baseline=np.zeros(min(600,len(result["beta_draws"]))); scenario=baseline.copy()
    for c in result["channels"]: baseline+=response_samples(result["current_spend"][c],c,result,len(baseline)); scenario+=response_samples(plan[c],c,result,len(baseline))
    return scenario-baseline
def optimize_budget(total_budget,minimums,maximums,result):
    channels=result["channels"]; lower=np.array([minimums[c] for c in channels]); upper=np.array([maximums[c] for c in channels])
    if lower.sum()>total_budget or upper.sum()<total_budget: raise ValueError("Budget must fall between the sum of channel minimums and maximums.")
    def objective(x): return -sum(response_samples(x[i],c,result,300).mean() for i,c in enumerate(channels))
    x0=np.clip(np.array([result["current_spend"][c] for c in channels]),lower,upper); x0+= (total_budget-x0.sum())/len(x0); x0=np.clip(x0,lower,upper)
    solution=minimize(objective,x0,method="SLSQP",bounds=list(zip(lower,upper)),constraints={"type":"eq","fun":lambda x:x.sum()-total_budget},options={"maxiter":200})
    if not solution.success: raise ValueError(solution.message)
    allocation={c:float(solution.x[i]) for i,c in enumerate(channels)}; return allocation,scenario_lift(allocation,result)
