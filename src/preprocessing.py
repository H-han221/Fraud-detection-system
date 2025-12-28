import pandas as pd
import ipaddress


def load_data(fraud_path, ip_path):
    fraud_df = pd.read_csv( "../data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("../data/raw/IpAddress_to_Country.csv")
    return fraud_df, ip_df


def clean_fraud_data(df):
    df = df.drop_duplicates()

    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Drop rows with missing critical values
    df = df.dropna(subset=[
        'user_id', 'purchase_value', 'ip_address', 'class'
    ])

    return df


def ip_to_int(ip):
    try:
        return int(float(ip))
    except (ValueError, TypeError):
        return None
def prepare_ip_data(ip_df):
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    return ip_df



def merge_ip_country(fraud_df, ip_df):
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    fraud_df = fraud_df.dropna(subset=['ip_int'])
    fraud_df['ip_int'] = fraud_df['ip_int'].astype(int)

    ip_df = ip_df.sort_values('lower_bound_ip_address')

    merged = pd.merge_asof(
        fraud_df.sort_values('ip_int'),
        ip_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    merged = merged[
        merged['ip_int'] <= merged['upper_bound_ip_address']
    ]

    return merged
