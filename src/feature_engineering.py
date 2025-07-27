import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_time_features(df, signup_col='signup_time', purchase_col='purchase_time'):
    """
    Extract useful time-based features.
    """
    logging.info("Extracting time features...")
    df[signup_col] = pd.to_datetime(df[signup_col])
    df[purchase_col] = pd.to_datetime(df[purchase_col])
    df['hour_of_day'] = df[purchase_col].dt.hour
    df['day_of_week'] = df[purchase_col].dt.dayofweek
    df['time_since_signup'] = (df[purchase_col] - df[signup_col]).dt.total_seconds() / 60
    logging.info("Time features extracted.")
    return df


def map_ip_to_country(fraud_df, ip_country_df):
    """
    Map IP integers to countries using fast merge_asof.
    """
    logging.info("Mapping IP addresses to countries...")
    fraud_df = fraud_df.sort_values('ip_int')
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')

    merged_df = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    merged_df['country'] = merged_df.apply(
        lambda row: row['country'] if row['ip_int'] <= row['upper_bound_ip_address'] else 'Unknown',
        axis=1
    )
    logging.info("IP to country mapping complete.")
    return merged_df


def transaction_frequency_features(df, user_col='user_id', time_col='purchase_time', window='1D'):
    """
    Calculate transaction frequency features per user in rolling windows.
    Fixes bug by using groupby with DatetimeIndex for rolling time windows.
    """
    logging.info(f"Calculating transaction frequency over rolling window '{window}'...")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([user_col, time_col])

    results = []
    for user, group in df.groupby(user_col):
        group = group.set_index(time_col).sort_index()
        group['txn_freq_last_window'] = group[user_col].rolling(window).count()
        results.append(group.reset_index())

    df_out = pd.concat(results).sort_index()
    logging.info("Transaction frequency features calculated.")
    return df_out


def encode_categorical_features(df, categorical_cols, drop_first=True):
    """
    One-hot encode specified categorical columns with option to drop first category.
    """
    logging.info(f"Encoding categorical columns: {categorical_cols} with drop_first={drop_first}...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    logging.info("Categorical encoding complete.")
    return df_encoded


def scale_numeric_features(df, numeric_cols, scaler=None):
    """
    Scale numeric columns using a provided scaler.
    If scaler is None, fit a new StandardScaler.
    Returns scaled df and scaler used (for reuse).
    """
    logging.info(f"Scaling numeric columns: {numeric_cols}...")
    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logging.info("Fitted new scaler and transformed data.")
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        logging.info("Used provided scaler to transform data.")
    return df, scaler


# ---------------------------------------------
# Advanced Feature Engineering Enhancements
# ---------------------------------------------

def add_time_behavior_features(df, user_col='user_id', time_col='purchase_time'):
    """
    Add features about timing behavior of transactions.
    """
    logging.info("Adding time behavior features...")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([user_col, time_col])

    df['time_since_last_txn'] = df.groupby(user_col)[time_col].diff().dt.total_seconds().fillna(-1)

    df['avg_time_between_txns'] = (
        df.groupby(user_col)['time_since_last_txn']
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # Rolling counts require setting DatetimeIndex per group, so:
    txn_count_hour = []
    txn_count_day = []
    for user, group in df.groupby(user_col):
        group = group.set_index(time_col).sort_index()
        txn_count_hour.append(group[user_col].rolling('1H').count().reset_index(drop=True))
        txn_count_day.append(group[user_col].rolling('1D').count().reset_index(drop=True))

    df['txn_count_last_hour'] = pd.concat(txn_count_hour).sort_index()
    df['txn_count_last_day'] = pd.concat(txn_count_day).sort_index()

    logging.info("Time behavior features added.")
    return df


def add_transaction_amount_features(df, user_col='user_id', amount_col='purchase_value'):
    """
    Add statistical features based on transaction amounts.
    """
    logging.info("Adding transaction amount features...")
    df['avg_purchase_amount'] = df.groupby(user_col)[amount_col].transform('mean')
    df['std_purchase_amount'] = df.groupby(user_col)[amount_col].transform('std').fillna(0)
    df['max_purchase_amount'] = df.groupby(user_col)[amount_col].transform('max')
    df['min_purchase_amount'] = df.groupby(user_col)[amount_col].transform('min')
    df['range_purchase_amount'] = df['max_purchase_amount'] - df['min_purchase_amount']
    logging.info("Transaction amount features added.")
    return df


def flag_geolocation_mismatch(df, signup_country_col='signup_country', purchase_country_col='country'):
    """
    Create feature for mismatched signup and transaction countries.
    """
    logging.info("Flagging geolocation mismatches...")
    df['geo_mismatch'] = (df[signup_country_col] != df[purchase_country_col]).astype(int)
    logging.info("Geo mismatch flag created.")
    return df


def device_behavior_features(df, user_col='user_id'):
    """
    Add features for device/browser/OS behavior.
    Assumes 'device', 'browser', and 'os' columns exist.
    """
    logging.info("Adding device behavior features...")
    df['device_browser_mismatch'] = (df['device_id'] != df['browser']).astype(int)
    df['os_change_flag'] = df.groupby(user_col)['os'].transform(lambda x: x != x.shift()).astype(int)
    logging.info("Device behavior features added.")
    return df


def frequency_encode(df, column):
    """
    Add frequency encoding for a categorical column.
    """
    logging.info(f"Frequency encoding for column: {column}")
    freq = df[column].value_counts(normalize=True)
    df[column + '_freq'] = df[column].map(freq)
    logging.info("Frequency encoding complete.")
    return df


if __name__ == "__main__":
    logging.info("Feature engineering module loaded.")
