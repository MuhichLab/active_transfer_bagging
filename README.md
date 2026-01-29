# Active-Transfer Bagging

Associated Article  -  https://arxiv.org/[arxiv-id when submitted]

# Install
ATBagging can be installed using pip via  

```
pip install "git+https://github.com/MuhichLab/transactive_learning.git"
```

# Dependencies
The dependencies include standard scientific python packages, i.e. numpy, scikit-learn, and scipy, along with the 
nonstandard [DPPy package](https://github.com/guilgautier/DPPy) to provide basic DPP sampling functionality.

# Examples
A working example with data is provided in the Examples/ directory.

The most basic functionality can be summarized in the following few lines.
With a dataset expressed as numpy ndarrays `X` and `y`, and a test set called `Xstar`:  

```
atb = ATBagging(n_estimators=100, random_seed=1234)
atb.fit(X, y)
downselection_results = atb.downselect(n=30, Xstar=Xstar)
indices = downselection_results.indices
```
