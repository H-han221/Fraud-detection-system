import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import clean_fraud_data, merge_ip_country
from src.feature_engineering import add_time_features, add_transaction_velocity
from src.evaluation import evaluate_model

# Dummy fraud data
def dummy_fraud_data():
    return pd.DataFrame({
        "user_id": [1, 2, 1, 3],
        "signup_time": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-01", "2025-01-03"
        ]),
        "purchase_time": pd.to_datetime([
            "2025-01-01 01:00", "2025-01-02 05:00", "2025-01-01 03:00", "2025-01-03 10:00"
        ]),
        "purchase_value": [100, 200, 150, 300],
        "device_id": ["d1", "d2", "d1", "d3"],
        "source": ["SEO", "Ads", "SEO", "SEO"],
        "browser": ["Chrome", "Safari", "Chrome", "Firefox"],
        "sex": ["M", "F", "M", "F"],
        "age": [25, 30, 25, 40],
        "ip_address": ["1.1.1.1", "2.2.2.2", "1.1.1.1", "3.3.3.3"],
        "class": [0, 1, 0, 1]
    })

# Dummy IP-to-country data
def dummy_ip_data():
    return pd.DataFrame({
        "lower_bound_ip_address": [16777216, 33554432, 50331648],
        "upper_bound_ip_address": [33554431, 50331647, 67108863],
        "country": ["US", "UK", "CA"]
    })

@pytest.mark.slow
def test_pipeline_end_to_end():
    # 1️⃣ Load dummy data
    df = dummy_fraud_data()
    ip_df = dummy_ip_data()

    # 2️⃣ Preprocessing
    df_clean = clean_fraud_data(df)
    df_merged = merge_ip_country(df_clean, ip_df)
    assert "country" in df_merged.columns

    # 3️⃣ Feature engineering
    df_feat = add_time_features(df_merged)
    df_feat = add_transaction_velocity(df_feat)
    assert "hour_of_day" in df_feat.columns
    assert "tx_count_24h" in df_feat.columns

    # 4️⃣ Minimal train/test split
    X = df_feat[["purchase_value", "hour_of_day", "day_of_week", "tx_count_24h"]]
    y = df_feat["class"]

    # 5️⃣ Train small models
    lr = LogisticRegression()
    lr.fit(X, y)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # 6️⃣ Evaluate
    lr_res = evaluate_model(lr, X, y)
    rf_res = evaluate_model(rf, X, y)

    # 7️⃣ Assertions
    for res in [lr_res, rf_res]:
        assert "F1" in res
        assert "AUC_PR" in res
        assert "Confusion_Matrix" in res
        # F1 should be between 0 and 1
        assert 0 <= res["F1"] <= 1
        # AUC-PR should be between 0 and 1
        assert 0 <= res["AUC_PR"] <= 1
