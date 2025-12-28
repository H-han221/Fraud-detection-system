import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_fraud_data, merge_ip_country
from src.feature_engineering import add_time_features, add_transaction_velocity

# Minimal dummy data
def dummy_fraud_data():
    return pd.DataFrame({
        "user_id": [1, 2, 1],
        "signup_time": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
        "purchase_time": pd.to_datetime(["2025-01-01 01:00", "2025-01-02 05:00", "2025-01-01 03:00"]),
        "purchase_value": [100, 200, 150],
        "device_id": ["d1", "d2", "d1"],
        "source": ["SEO", "Ads", "SEO"],
        "browser": ["Chrome", "Safari", "Chrome"],
        "sex": ["M", "F", "M"],
        "age": [25, 30, 25],
        "ip_address": ["1.1.1.1", "2.2.2.2", "1.1.1.1"],
        "class": [0, 1, 0]
    })

def dummy_ip_data():
    return pd.DataFrame({
        "lower_bound_ip_address": [16777216, 33554432],
        "upper_bound_ip_address": [33554431, 50331647],
        "country": ["US", "UK"]
    })

def test_clean_fraud_data():
    df = dummy_fraud_data()
    df_clean = clean_fraud_data(df)
    assert "purchase_value" in df_clean.columns
    assert df_clean.shape[0] == 3

def test_merge_ip_country():
    df = dummy_fraud_data()
    ip_df = dummy_ip_data()
    merged = merge_ip_country(df, ip_df)
    # Test column exists
    assert "country" in merged.columns
    # Values not null
    assert merged["country"].notnull().all()

def test_add_time_features():
    df = dummy_fraud_data()
    df_time = add_time_features(df)
    assert "hour_of_day" in df_time.columns
    assert "day_of_week" in df_time.columns
    assert "time_since_signup" in df_time.columns

def test_add_transaction_velocity():
    df = add_time_features(dummy_fraud_data())
    df_vel = add_transaction_velocity(df)
    assert "tx_count_24h" in df_vel.columns
    # Must be at least 1
    assert (df_vel["tx_count_24h"] >= 1).all()
