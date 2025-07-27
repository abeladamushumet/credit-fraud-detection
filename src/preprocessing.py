import pandas as pd
import numpy as np

def clean_fraud_data(fraud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the e-commerce fraud data.
    - Handle missing values
    - Fix data types
    - Remove duplicates
    - Basic sanity checks
    
    Args:
        fraud_df: Raw fraud data DataFrame
    
    Returns:
        Cleaned fraud data DataFrame
    """
    # Drop duplicates
    fraud_df = fraud_df.drop_duplicates()
    
    # Handle missing values - example: drop rows with missing target or critical features
    fraud_df = fraud_df.dropna(subset=['class', 'ip_address', 'purchase_time', 'signup_time'])
    
    # Fill other missing values or choose strategy depending on features
    fraud_df['browser'] = fraud_df['browser'].fillna('unknown')
    fraud_df['source'] = fraud_df['source'].fillna('unknown')
    fraud_df['sex'] = fraud_df['sex'].fillna('U')  # U for unknown
    
    # Fix data types
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce')
    
    # Convert IP address if stored as float - keep as int
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    
    # Convert target/class column to int
    fraud_df['class'] = fraud_df['class'].astype(int)
    
    # Drop rows with invalid purchase or signup time after conversion
    fraud_df = fraud_df.dropna(subset=['purchase_time', 'signup_time'])
    
    return fraud_df.reset_index(drop=True)


def clean_creditcard_data(creditcard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the credit card transactions data.
    - Handle missing values
    - Fix data types
    - Remove duplicates
    
    Args:
        creditcard_df: Raw credit card data DataFrame
    
    Returns:
        Cleaned credit card data DataFrame
    """
    creditcard_df = creditcard_df.drop_duplicates()
    
    # Check and drop missing target
    creditcard_df = creditcard_df.dropna(subset=['Class'])
    
    # Ensure 'Class' is int type
    creditcard_df['Class'] = creditcard_df['Class'].astype(int)
    
    # Check for missing values in features (V1-V28, Time, Amount)
    # You can fill or drop depending on analysis - typically this dataset is clean
    creditcard_df = creditcard_df.dropna()
    
    return creditcard_df.reset_index(drop=True)


def clean_ip_country_data(ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean IP-to-country mapping data.
    - Rename columns if necessary
    - Convert IP address bounds to integers (if not already)
    - Remove duplicates
    
    Args:
        ip_country_df: Raw IP-country mapping DataFrame
    
    Returns:
        Cleaned IP-country DataFrame
    """
    ip_country_df.columns = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    
    # If IP bounds are strings  convert to int
    # Otherwise, ensure integer type
    if ip_country_df['lower_bound_ip_address'].dtype == 'O':  # Object/string type
        import ipaddress
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    else:
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(int)
        ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(int)
    
    ip_country_df = ip_country_df.drop_duplicates()
    
    return ip_country_df.reset_index(drop=True)
