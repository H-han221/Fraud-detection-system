from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model on test data.
    
    Works with sparse matrices or DataFrames.
    Returns F1, AUC-PR, and confusion matrix.
    """
    # Convert X_test to dense if sparse
    if hasattr(X_test, "toarray"):
        X_test_eval = X_test.toarray()
    else:
        X_test_eval = X_test

    # Predictions
    y_pred = model.predict(X_test_eval)
    y_proba = model.predict_proba(X_test_eval)[:, 1]

    # Metrics
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "F1": f1,
        "AUC_PR": auc_pr,
        "Confusion_Matrix": cm

    }
from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    pr_auc = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    
    return {
        "f1_mean": f1.mean(),
        "f1_std": f1.std(),
        "pr_auc_mean": pr_auc.mean(),
        "pr_auc_std": pr_auc.std()
    }

