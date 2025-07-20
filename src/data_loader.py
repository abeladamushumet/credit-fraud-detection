import pandas as pd
from typing import Tuple

def load_fraud_data(path: str) -> pd.DataFrame:
    """Load the Fraud_Data.csv e-commerce dataset."""
    df = pd.read_csv(path)
    print(f"Loaded Fraud Data: {df.shape} rows, columns: {df.columns.tolist()}")
    return df

def load_creditcard_data(path: str) -> pd.DataFrame:
    """Load the creditcard.csv bank transactions dataset."""
    df = pd.read_csv(path)
    print(f"Loaded Credit Card Data: {df.shape} rows, columns: {df.columns.tolist()}")
    return df

def load_ip_country_data(path: str) -> pd.DataFrame:
    """Load the IP address to country mapping dataset."""
    df = pd.read_csv(path)
    print(f"Loaded IP-Country Data: {df.shape} rows, columns: {df.columns.tolist()}")
    return df

def load_all_data(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all raw datasets and return as tuple.
    
    Args:
        raw_dir: Path to the raw data folder.
    
    Returns:
        fraud_df, creditcard_df, ip_country_df
    """
    fraud_path = f"{raw_dir}\\Fraud_Data.csv"
    creditcard_path = f"{raw_dir}\\creditcard.csv"
    ip_country_path = f"{raw_dir}\\IpAddress_to_Country.csv"
    
    fraud_df = load_fraud_data(fraud_path)
    creditcard_df = load_creditcard_data(creditcard_path)
    ip_country_df = load_ip_country_data(ip_country_path)
    
    return fraud_df, creditcard_df, ip_country_df


if __name__ == "__main__":

    raw_data_dir = r"C:\Users\hp\Desktop\10 Acadamy\VS code\credit-fraud-detection\data\raw"
    fraud_df, creditcard_df, ip_country_df = load_all_data(raw_data_dir)
