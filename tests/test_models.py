import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_model

def dummy_features_labels():
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y = np.array([0, 1, 0, 1])
    return X, y

def test_logistic_regression():
    X, y = dummy_features_labels()
    lr = LogisticRegression()
    lr.fit(X, y)
    results = evaluate_model(lr, X, y)
    assert "F1" in results
    assert "AUC_PR" in results
    assert "Confusion_Matrix" in results

def test_random_forest():
    X, y = dummy_features_labels()
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    results = evaluate_model(rf, X, y)
    assert "F1" in results
    assert "AUC_PR" in results
    assert "Confusion_Matrix" in results
