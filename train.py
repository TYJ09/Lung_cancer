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

data_dir_path = os.getcwd()

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
    cols = list(X_processed.columns)
    with open(f'{data_dir_path}/features.txt', 'w') as fp:
      fp.write("\n".join(cols))
    return train_test_split(X_processed, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'NaiveBayes': GaussianNB()
}

results = {}




for name, model in models.items():

    print(f'Processing started for model {name}...')
    print(param_grid[name])

    #if not os.path.exists(f'{data_dir_path}/{name}_model.pkl'):
    grid = GridSearchCV(model, param_grid[name]['params'], cv=10, scoring='accuracy', n_jobs=-1)
    X_train, X_test, y_train, y_test = get_train_test_data(param_grid[name]['ohe'], param_grid[name]['standardize'])
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    # Save the model
    joblib.dump(grid.best_estimator_, f'{data_dir_path}/{name}_model.pkl')


    # Predictions
    y_pred = grid.best_estimator_.predict(X_test)
    #y_pred = np.array(y_pred)
    #y_test = np.array(y_test)


        # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results[name] = {
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist()
    }

        # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{data_dir_path}/{name}_confusion_matrix.png')

    print(f'Processing completed for model {name}...')

# Save results to JSON
with open(f'{data_dir_path}/model_evaluation_results.json', 'w') as file:
    json.dump(results, file, indent=4)
results.open
