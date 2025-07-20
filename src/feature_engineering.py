import pandas as pd
import numpy as np

def extract_time_features(df, signup_col='signup_time', purchase_col='purchase_time'):
    """
    Extract useful time-based features:
    - hour_of_day (from purchase time)
    - day_of_week (from purchase time)
    - time_since_signup (difference in seconds or minutes)
    """

    # Convert columns to datetime if not already
    df[signup_col] = pd.to_datetime(df[signup_col])
    df[purchase_col] = pd.to_datetime(df[purchase_col])

    # Hour of day when purchase happened
    df['hour_of_day'] = df[purchase_col].dt.hour

    # Day of the week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[purchase_col].dt.dayofweek

    # Time difference between signup and purchase in minutes
    df['time_since_signup'] = (df[purchase_col] - df[signup_col]).dt.total_seconds() / 60

    return df


def map_ip_to_country(fraud_df, ip_country_df):
    """
    Map IP integers to countries using fast merge_asof.

    Assumes:
    - fraud_df has 'ip_int' column (integer representation of IP)
    - ip_country_df has 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'
    """

    fraud_df = fraud_df.sort_values('ip_int')
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')

    merged_df = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Check if ip_int falls within upper bound
    merged_df['country'] = merged_df.apply(
        lambda row: row['country'] if row['ip_int'] <= row['upper_bound_ip_address'] else 'Unknown',
        axis=1
    )

    return merged_df


def transaction_frequency_features(df, user_col='user_id', time_col='purchase_time', window='1D'):
    """
    Calculate transaction frequency features per user in rolling windows.
    Example: number of transactions in the last 1 day ('1D') for each transaction.

    Parameters:
    - df: DataFrame with transaction data
    - user_col: user identifier column
    - time_col: transaction timestamp column (must be datetime)
    - window: rolling window size, e.g., '1D' for one day

    Returns:
    - df with new feature: 'txn_freq_last_1D'
    """

    df = df.sort_values([user_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col])

    # Rolling count of transactions per user
    df['txn_freq_last_1D'] = df.groupby(user_col)[time_col].rolling(window).count().reset_index(level=0, drop=True)

    return df


def encode_categorical_features(df, categorical_cols):
    """
    One-hot encode specified categorical columns.
    """

    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def scale_numeric_features(df, numeric_cols, scaler):
    """
    Scale numeric columns using a provided scaler (e.g., StandardScaler, MinMaxScaler).
    """

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


if __name__ == "__main__":

    pass
