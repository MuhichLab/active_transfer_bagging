# Image spot hold
<img src="ASAP_logo.png" align="middle" />

Associated Journal Article  -  https://XXXXXXXXXXXXXXXXXXXXXX

# Install
ATBagging can be installed using pip via
```
pip install "git+https://github.com/MuhichLab/transactive_learning.git"
```

# Dependencies
The dependencies include standard scientific python packages, i.e.
- numpy
- scikit-learn
- scipy
As well as the DPP sampling functionality provided by the [DPPy package](https://github.com/guilgautier/DPPy)

# Examples
With a dataset expressed as numpy ndarrays `X` and `y`, and a test set called `Xstar`:
```
atb = ATBagging(n_estimators=100, random_seed=1234)
atb.fit(X, y)
downselection_results = atb.downselect(n=30, Xstar=Xstar)
indices = downselection_results.indices
```
