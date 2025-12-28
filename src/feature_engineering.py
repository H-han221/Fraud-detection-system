import pandas as pd


def add_time_features(df):
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600

    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    return df

import pandas as pd

import pandas as pd

def add_transaction_velocity(df):
    # Ensure datetime
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Sort data
    df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)

    # Compute rolling transaction count
    tx_counts = (
        df.groupby('user_id')
          .rolling('24H', on='purchase_time')['purchase_time']
          .count()
          .reset_index(drop=True)
    )

    # Assign safely (no index alignment)
    df['tx_count_24h'] = tx_counts.values

    # Fill missing values
    df['tx_count_24h'] = df['tx_count_24h'].fillna(1)

    return df
