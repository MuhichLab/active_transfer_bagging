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

def load_data(file="qm9_full.csv", split=0.2, state=None):
    """
    Returns: X_train, X_test, y_train, y_test (all numpy arrays)
    - Supports .pkl (tuple (X, Y) with Y[:,0] used) and .csv
    - Detects LFS pointer files and raises an actionable error.
    """
    if file.endswith(".pkl"):
        with open(file, "rb") as f:
            X, Y = pickle.load(f)
        y = Y[:, 0]  # your original convention
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=state)
        return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

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

    # Special handling for qm9_full.csv
    if os.path.basename(path) == "qm9_full.csv":
        df = pd.read_csv(path)
        X = df.iloc[:, :-4].copy()
        y = df.iloc[:, -4].copy()  # target U0
        if "id_cat" not in X.columns:
            raise ValueError("Expected 'id_cat' column for stratify in qm9_full.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=state, stratify=X["id_cat"]
        )
        # Drop non-feature columns post split
        for set_ in (X_train, X_test):
            for col in ("system", "id_cat"):
                if col in set_.columns:
                    set_.drop(columns=[col], inplace=True)
        X_train = _to_numeric_matrix(X_train)
        X_test  = _to_numeric_matrix(X_test)
        y_train = _prep_target(y_train)
        y_test  = _prep_target(y_test)
        return X_train.to_numpy(), X_test.to_numpy(), y_train, y_test

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
