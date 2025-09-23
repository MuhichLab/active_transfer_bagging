#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_data

random_state = 42

# ---------------------- STYLE (publication-ish) ----------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
})

# ---------------------- Helpers ----------------------
def display_name(path):
    """Pretty title without .csv, underscores to spaces, Title Case."""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return re.sub(r'\s+', ' ', name.replace('_', ' ')).title()

def family_key(path):
    """
    Group datasets that share the same 'family' name.
    Heuristic:
      - strip extension
      - if it ends with _<digits>, drop the numeric suffix
      - otherwise keep as-is
    Ex: weather_1 -> weather ; weather_2 -> weather ; pm25 -> pm25
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(.*?)(?:_\d+)?$', stem)  # non-greedy, drop _<digits> if present
    fam = m.group(1) if m else stem
    return fam.lower()

def family_label(fam):
    """Left-margin label for a family key, prettified."""
    return fam.replace('_', ' ').title()

# ---------------------- Data discovery ----------------------
csv_files = glob.glob("./Datasets/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found in ./Datasets/*.csv")

# Sort by (family, then name) so rows from same family stay together
csv_files = sorted(csv_files, key=lambda p: (family_key(p), os.path.basename(p).lower()))
n_rows = len(csv_files)

# ---------------------- Model / tuning config ----------------------
search_space = {
    'n_estimators': (100, 500),
    'max_depth': (1, 30),
    'min_samples_leaf': (1, 20),
}
tune_bays = False
base_forest = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=random_state)

# ---------------------- Figure layout ----------------------
fig, axs = plt.subplots(n_rows, 2, figsize=(8.8, 3.4 * n_rows),
                        gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
if n_rows == 1:
    axs = np.expand_dims(axs, axis=0)  # ensure 2D

# Draw subtle family labels in the left margin at the start of each group
y_offsets = []
families = [family_key(f) for f in csv_files]
for i, fam in enumerate(families):
    if i == 0 or fam != families[i-1]:
        y_offsets.append((i, fam))

# ---------------------- Main loop ----------------------
for i, file in enumerate(csv_files):
    # Load data
    X_train, X_test, y_train, y_test = load_data(file=file, split=0.2, state=random_state)

    # Choose model
    if tune_bays:
        opt = BayesSearchCV(
            estimator=base_forest,
            search_spaces=search_space,
            scoring='neg_mean_absolute_error',
            cv=5, n_iter=64, n_points=4, n_jobs=-1,
            verbose=0, random_state=42
        )
        opt.fit(X_train, y_train)
        model = opt.best_estimator_
        print(f"Best params for {os.path.basename(file)}: {opt.best_params_}")
    else:
        model = base_forest

    # --- Fit & predict ---
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, pred_train)
    r2_test  = r2_score(y_test,  pred_test)

    # Axis limits shared per row (so train/test have same scale)
    all_vals = np.concatenate([y_train, pred_train, y_test, pred_test])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad

    # ---------------------- Train panel ----------------------
    axL = axs[i, 0]
    axL.scatter(y_train, pred_train, s=16, alpha=0.6, edgecolor='none')
    axL.plot([lo, hi], [lo, hi], ls='--', c='k', lw=1.2)
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
    axL.set_xlabel("Actual")
    axL.set_ylabel("Predicted")
    axL.set_title(f"Train – {display_name(file)}", pad=10)

    # Replace RMSE box with R²
    axL.text(0.04, 0.96, f"$R^2$ = {r2_train:.2f}",
             transform=axL.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9))

    # ---------------------- Test panel ----------------------
    axR = axs[i, 1]
    axR.scatter(y_test, pred_test, s=16, alpha=0.6, edgecolor='none')
    axR.plot([lo, hi], [lo, hi], ls='--', c='k', lw=1.2)
    axR.set_xlim(lo, hi); axR.set_ylim(lo, hi)
    axR.set_xlabel("Actual")
    axR.set_ylabel("Predicted")
    axR.set_title(f"Test – {display_name(file)}", pad=10)

    axR.text(0.04, 0.96, f"$R^2$ = {r2_test:.2f}",
             transform=axR.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9))


plt.tight_layout()

# ---------------------- Save high-res outputs ----------------------
out_png = "rf_scatter_train_test.png"
out_pdf = "rf_scatter_train_test.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print(f"Saved: {out_png}, {out_pdf}")
