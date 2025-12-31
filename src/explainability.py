# src/explainability.py
import shap
import numpy as np

def shap_explain_tree(model, X, feature_names=None, sample_size=2000):
    """
    Generate SHAP values for tree-based models (Random Forest).
    Returns:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        X_sample: subsampled feature matrix
        feature_names: feature names (same as input)
    """
    X = np.array(X)  # ensure numpy array

    # Subsample for speed
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Ensure feature names is numpy array
    if feature_names is not None:
        feature_names = np.array(feature_names)

    return explainer, shap_values, X_sample, feature_names

def shap_explain_logistic(model, X, feature_names=None, sample_size=2000):
    """
    Generate SHAP values for linear/logistic models.
    Returns: explainer, shap_values, X_sample, feature_names
    """
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample, feature_names

