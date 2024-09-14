#%%

import pandas as pd
import seaborn as sns
import pymc as pm
import bambi as bmb
import arviz as az
#%%

dat = sns.load_dataset('penguins').dropna()
# %%
model = bmb.Model('body_mass_g ~ bill_length_mm + bill_depth_mm + (1|island)', data = dat, dropna=True)
fitted = model.fit()
# %%
# make pymc5 model to make predictions
# body_mass_g ~ bill_length_mm + bill_depth_mm + (1 | island)

with pm.Model() as model:
    w0 = pm.Normal("w0", mu=0, sigma=1)
    w1 = pm.Normal("w1", mu=0, sigma=1)
    w2 = pm.Normal("w2", mu=0, sigma=1)
    epsilon = pm.HalfCauchy("epsilon", 5)
    mu = pm.Deterministic("mu", w0 + w1 * dat["bill_length_mm"] + w2 * dat["bill_depth_mm"])
    y_pred = pm.Normal("y_pred", mu=mu, sigma=epsilon, observed=dat["body_mass_g"])

    trace = pm.sample(1000, chains=2, cores=1)

az.plot_trace(trace, var_names=["w0", "w1", "w2"])
az.plot_autocorr(trace, var_names=["w0", "w1", "w2"], combined=True)
az.summary(trace, var_names=["w0", "w1", "w2"])


# %%
from pymc import model_to_graphviz
model_to_graphviz(model)
# %%
