#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
import matplotlib.pyplot as plt
import json
import joblib
import seaborn as sns
import sys

data_dir_path = os.listdir()

if len(sys.argv) != 2:
    print('Usage: python inference.py <data_dir_path>')
    sys.exit(1)

target_column = 'Lung Cancer Occurrence'
merged_df = pd.read_csv(sys.argv[-1])

X = merged_df.drop(target_column, axis=1)
y = merged_df[target_column]

y_binarized = label_binarize(y, classes=np.unique(y))

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

with open('/content/drive/MyDrive/Lung_cancer/config.json', 'r') as file:
    param_grid = json.load(file)

def apply_ohe(X):
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_encoded = ohe.fit_transform(X[categorical_cols]).toarray()
    joblib.dump(ohe, f'{data_dir_path}/ohe.pkl')
    return pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorical_cols))

def apply_standardization(X, X_encoded=None):
    if X_encoded is not None:
        X_numerical = X[numerical_cols].reset_index(drop=True)
        X_combined = pd.concat([X_numerical, X_encoded], axis=1)
    else:
        X_combined = X[numerical_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    joblib.dump(scaler, f'{data_dir_path}/scaler.pkl')
    return pd.DataFrame(X_scaled, columns=X_combined.columns)

def get_train_test_data(ohe_required, normalization_required):
    X_processed = X.copy()
    X_encoded = None
    if ohe_required:
        X_encoded = apply_ohe(X)
        X_processed = pd.concat([X[numerical_cols], X_encoded], axis=1)
    if normalization_required:
        X_processed = apply_standardization(X, X_encoded)
    return X_processed

if __name__ == '__main__':

    rf_estimator = param_grid['RandomForest']
    X_test = get_train_test_data(rf_estimator['ohe'], rf_estimator['standardize'])
    model = joblib.load(f'{data_dir_path}/RandomForest_model.pkl')
    results = {}
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y, y_pred)
    class_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    results['RandomForest'] = {
                'Accuracy': accuracy,
                'Classification Report': class_report,
                'Confusion Matrix': conf_matrix.tolist()
    }
    print(results)