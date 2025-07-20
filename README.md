# 💳 Credit Card Fraud Detection Using Machine Learning

This project builds a robust machine learning pipeline to detect fraudulent credit card transactions using real-world data. It is designed for high accuracy, scalability, and integration into real-time financial applications.

---

## 🚀 Project Overview

- **Client**: Adey Innovations Inc.
- **Objective**: Detect fraudulent transactions in e-commerce and banking environments.
- **Tech Stack**: Python, Scikit-learn, Pandas, XGBoost, SHAP, Streamlit


---

## 📁 Directory Structure

```bash
credit-fraud-detection/
│
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for EDA and modeling
├── src/                      # Scripts for preprocessing, training, and evaluation
├── reports/                  # Interim and final project reports
├── models/                   # Saved ML models
├── app/                      # Streamlit web app
├── requirements.txt          # All Python dependencies
├── .gitignore
├── README.md                 # You're here!
└── interim_report.txt        # Initial findings and recommendations

---

## 🔍 Datasets

Located in `data/raw/`:
- `Fraud_Data.csv` — E-commerce transaction records
- `creditcard.csv` — Bank credit card transaction data (from Kaggle)
- `IpAddress_to_Country.csv` — IP geolocation mapping

Processed versions are stored in `data/processed/`.

---

## 🛠 Features

- ✅ **Cleaning & Preprocessing:** Nulls, types, outliers
- ✅ **Feature Engineering:**
  - Time-based: transaction hour, day
  - Frequency-based: user activity patterns
  - Geo-based: country-level risk
- ✅ **Imbalance Handling:** SMOTE, undersampling
- ✅ **Modeling:** Logistic Regression, Random Forest, XGBoost
- ✅ **Evaluation:** AUC-PR, F1-score, Confusion Matrix
- ✅ **Explainability:** SHAP values for model transparency

---

## 📓 Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `1_data_cleaning.ipynb` | Prepare raw data |
| `2_eda_fraud_data.ipynb` | Visualize e-commerce transactions |
| `3_eda_creditcard.ipynb` | Analyze bank transaction data |
| `4_feature_engineering.ipynb` | Time/geo/frequency features |
| `5_model_training.ipynb` | Train ML models |
| `6_model_evaluation.ipynb` | Assess performance |
| `7_model_explainability.ipynb` | SHAP plots and insights |

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/abeladamushumet/fraud-detection-project.git
cd fraud-detection-project
pip install -r requirements.txt
