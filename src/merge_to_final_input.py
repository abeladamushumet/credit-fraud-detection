import pandas as pd
from pathlib import Path

# Define file paths
base_path = Path("data/processed")
fraud_file = base_path / "fraud_Data_features.csv"
credit_file = base_path / "creditcard_features.csv"

# Load datasets
fraud_df = pd.read_csv(fraud_file)
credit_df = pd.read_csv(credit_file)

# Rename credit_df target column for consistency
credit_df.rename(columns={"Class": "class"}, inplace=True)

# Function to move target column to the end for readability
def reorder_columns(df, target='class'):
    cols = list(df.columns)
    if target in cols:
        cols.remove(target)
        cols.append(target)
    return df[cols]

fraud_df = reorder_columns(fraud_df)
credit_df = reorder_columns(credit_df)

# Select top 10 country one-hot columns in fraud_df to reduce sparsity
country_cols = [col for col in fraud_df.columns if col.startswith("country_")]
top_country_cols = fraud_df[country_cols].sum().sort_values(ascending=False).head(10).index.tolist()

# Define columns to keep from fraud_df
fraud_columns = [
    "purchase_value", "age", "hour_of_day", "day_of_week", "time_since_signup",
    "sex_M", "source_Direct", "source_SEO",
    "browser_FireFox", "browser_IE", "browser_Opera", "browser_Safari"
] + top_country_cols + ["class"]

# Define columns to keep from credit_df
credit_columns = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "hour_of_day", "class"]

# Filter the dataframes by selected columns
fraud_df = fraud_df[fraud_columns]
credit_df = credit_df[credit_columns]

# Synchronize columns between both dfs by adding missing columns filled with 0
all_cols = set(fraud_df.columns).union(set(credit_df.columns))

for col in all_cols:
    if col not in fraud_df.columns:
        fraud_df[col] = 0
    if col not in credit_df.columns:
        credit_df[col] = 0

# Reorder columns consistently, with 'class' at the end
all_features = sorted(col for col in all_cols if col != "class")
final_columns = all_features + ["class"]

fraud_df = fraud_df[final_columns]
credit_df = credit_df[final_columns]

# Concatenate both dataframes vertically (fraud + credit)
final_df = pd.concat([fraud_df, credit_df], axis=0).reset_index(drop=True)

# Fill any remaining NaN values with 0
final_df.fillna(0, inplace=True)

# Save the final dataset for modeling
output_file = base_path / "final_model_input.csv"
final_df.to_csv(output_file, index=False)

print(f"[✅] Final model input saved to {output_file}")
