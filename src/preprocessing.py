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
 




def merge_ip_country(df, ip_df):
    # Convert IP string to integer if not done yet
    if df['ip_address_int'].dtype != np.int64:
        df['ip_address_int'] = df['ip_address_int'].astype(np.int64)

    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(np.int64)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(np.int64)

    # Sort by IP for merge_asof
    df = df.sort_values('ip_address_int')
    ip_df = ip_df.sort_values('lower_bound_ip_address')

    merged = pd.merge_asof(
        df,
        ip_df,
        left_on='ip_address_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    return merged

