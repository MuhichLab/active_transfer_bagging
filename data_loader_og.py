#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:06:26 2025

@author: djriver7
"""

import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Path to your CSV files (update this)
csv_files = glob.glob("/data/grp_cmuhich/carpetbagging/datasets/*.csv")

# Dictionary to store processed data
data_dict = {}

# Load, encode categorical columns, split, and scale data
for file in csv_files:
    filename = os.path.basename(file)
    if filename == 'qm9_full.csv':
        df = pd.read_csv(file)  # Load dataset
        
        # Using U0 as the target for now
        X = df.iloc[:, :-4]  # All columns except last four
        Y = df.iloc[:, -4]   # Fourth to last column as target

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, 
                                                stratify=X['id_cat'],random_state=42)

        for set_ in (X_train, X_test):
            set_.drop(columns=['system','id_cat'], axis=1,inplace=True)

    else:
        df = pd.read_csv(file)  # Load dataset
        
        # Assuming the last column is the target (Y)
        X = df.iloc[:, :-1]  # All columns except last
        Y = df.iloc[:, -1]   # Last column as target
        
        # Convert categorical target variable Y to numerical if necessary
        if Y.dtype == 'O':  # If object (categorical), try converting
            Y = pd.factorize(Y)[0]
        
        # Encode categorical columns in X
        for col in X.columns:
            if X[col].dtype == 'O':  # If column is categorical
                X[col] = pd.factorize(X[col])[0]  # Assign unique numbers to each category
        
        # Train-test split (80% train, 20% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # Scale X data (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store results
    data_dict[filename] = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "scaler": scaler  # Store scaler in case inverse transform is needed
    }
    
        # print(f"Processed {filename} - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    
# Now you can access data_dict['filename.csv'] for processed datasets.
with open('./weather_subsamp_Xy_train.pkl','rb') as f:
    X, Y = pickle.load(f)
y1 = Y[:,0]
y2 = Y[:,1]