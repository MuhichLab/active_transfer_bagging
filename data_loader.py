#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, glob, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LFS_SIGNS = ("version https://git-lfs.github.com/spec/v1", "oid sha256:")

def _is_lfs_pointer(path, n_lines=3):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join([next(f) for _ in range(n_lines)])
        return any(sig in head for sig in LFS_SIGNS)
    except Exception:
        return False

def _to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    # Convert non-numeric columns via one-hot encoding (drop_first to reduce collinearity)
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        X = pd.get_dummies(X, columns=non_num, drop_first=True)
    # Ensure purely numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Drop columns that became all-NaN after coercion
    X = X.dropna(axis=1, how="all")
    # Fill any residual NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))
    return X

def _prep_target(y: pd.Series) -> np.ndarray:
    # If y is non-numeric (object), factorize to integers
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    return y.to_numpy().reshape(-1)

def load_data(file, split=0.2, state=None):

    # CSV path
    matches = glob.glob(file)
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {file}")

    path = matches[0]
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"File appears to be a Git LFS pointer: {path}\n"
            f"Run:\n  git lfs install\n  git lfs pull\nThen re-run."
        )

    # Generic CSV: last column is target
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} has <2 columns (did you pull LFS content?): shape={df.shape}")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    X = _to_numeric_matrix(X)
    y = _prep_target(y)

    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError(f"Invalid shapes after preprocessing: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=split, random_state=state
    )
    return X_train, X_test, y_train, y_test
