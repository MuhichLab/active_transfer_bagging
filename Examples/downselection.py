import numpy as np, polars as pl
from ATBagging import ATBagging
cl = pl.col

random_seed = 1234
rng = np.random.default_rng(random_seed)

df = pl.read_parquet('data/pm25_X_y.parquet')[:1000]

feature_labels = df.columns[1:-1]
target_labels = df.columns[-1]

Xtr = df[feature_labels].to_numpy()
ytr = df[target_labels].to_numpy()

Xstar = Xtr[rng.choice(len(Xtr),300)]

atb = ATBagging(random_seed=1234)
atb.fit(Xtr,ytr)
downselection_results = atb.downselect(n=30, Xstar=Xstar, verbose=True)
indices = downselection_results.indices
