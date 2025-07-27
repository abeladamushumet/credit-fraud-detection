import streamlit as st
import pandas as pd
import joblib

# Load model once and cache
@st.cache_resource
def load_model():
    model = joblib.load("outputs/models/best_model.pkl")
    return model

model = load_model()

st.title("💳 Fraud Detection - Predict Transaction Fraud")

important_features = {
    "purchase_value": {"type": "number", "min": 0.0, "max": 10000.0, "default": 100.0},
    "age": {"type": "slider", "min": 18, "max": 100, "default": 30},
    "hour_of_day": {"type": "slider", "min": 0, "max": 23, "default": 12},
    "day_of_week": {"type": "slider", "min": 0, "max": 6, "default": 3},
    "time_since_signup": {"type": "number", "min": 0.0, "max": 1000.0, "default": 30.0},
    "sex_M": {"type": "selectbox", "options": [0, 1], "default": 1},
    "source_Direct": {"type": "selectbox", "options": [0, 1], "default": 0},
    "browser_FireFox": {"type": "selectbox", "options": [0, 1], "default": 0},
    "browser_IE": {"type": "selectbox", "options": [0, 1], "default": 0},
    "country_US": {"type": "selectbox", "options": [0, 1], "default": 0},
    "country_CA": {"type": "selectbox", "options": [0, 1], "default": 0},
    "Amount_scaled": {"type": "number", "min": 0.0, "max": 10.0, "default": 0.5},
}

st.write("Please fill in the transaction details:")

with st.form("fraud_detection_form"):
    user_input = {}
    with st.expander("Transaction Info", expanded=True):
        user_input["purchase_value"] = st.number_input(
            "Purchase Value",
            min_value=important_features["purchase_value"]["min"],
            max_value=important_features["purchase_value"]["max"],
            value=important_features["purchase_value"]["default"]
        )
        user_input["hour_of_day"] = st.slider(
            "Hour of Day",
            min_value=important_features["hour_of_day"]["min"],
            max_value=important_features["hour_of_day"]["max"],
            value=important_features["hour_of_day"]["default"]
        )
        user_input["day_of_week"] = st.slider(
            "Day of Week (0=Mon)",
            min_value=important_features["day_of_week"]["min"],
            max_value=important_features["day_of_week"]["max"],
            value=important_features["day_of_week"]["default"]
        )
        user_input["time_since_signup"] = st.number_input(
            "Time Since Signup (days)",
            min_value=important_features["time_since_signup"]["min"],
            max_value=important_features["time_since_signup"]["max"],
            value=important_features["time_since_signup"]["default"]
        )
        user_input["Amount_scaled"] = st.number_input(
            "Scaled Transaction Amount",
            min_value=important_features["Amount_scaled"]["min"],
            max_value=important_features["Amount_scaled"]["max"],
            value=important_features["Amount_scaled"]["default"]
        )

    with st.expander("User Demographics"):
        user_input["age"] = st.slider(
            "Age",
            min_value=important_features["age"]["min"],
            max_value=important_features["age"]["max"],
            value=important_features["age"]["default"]
        )
        user_input["sex_M"] = st.selectbox(
            "Sex (Male=1, Female=0)",
            options=important_features["sex_M"]["options"],
            index=important_features["sex_M"]["default"]
        )
        user_input["source_Direct"] = st.selectbox(
            "Source is Direct (Yes=1, No=0)",
            options=important_features["source_Direct"]["options"],
            index=important_features["source_Direct"]["default"]
        )

    with st.expander("Browser & Country"):
        user_input["browser_FireFox"] = st.selectbox(
            "Browser is Firefox (Yes=1, No=0)",
            options=important_features["browser_FireFox"]["options"],
            index=important_features["browser_FireFox"]["default"]
        )
        user_input["browser_IE"] = st.selectbox(
            "Browser is Internet Explorer (Yes=1, No=0)",
            options=important_features["browser_IE"]["options"],
            index=important_features["browser_IE"]["default"]
        )
        user_input["country_US"] = st.selectbox(
            "Country is US (Yes=1, No=0)",
            options=important_features["country_US"]["options"],
            index=important_features["country_US"]["default"]
        )
        user_input["country_CA"] = st.selectbox(
            "Country is Canada (Yes=1, No=0)",
            options=important_features["country_CA"]["options"],
            index=important_features["country_CA"]["default"]
        )

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    # Get model's expected features in order
    model_features = model.booster_.feature_name()

    # Create a dict with all features initialized to zero
    full_input = {feature: 0 for feature in model_features}

    for key, val in user_input.items():
        if key in full_input:
            full_input[key] = val

    # Convert to DataFrame with the correct column order
    input_df = pd.DataFrame([full_input], columns=model_features)

    # Prediction
    prediction_proba = model.predict_proba(input_df)[:, 1][0]
    prediction = model.predict(input_df)[0]

    st.markdown("### Prediction Results:")
    st.write(f"Fraud Probability: **{prediction_proba:.4f}**")
    st.write(f"Predicted Class: **{'Fraud' if prediction == 1 else 'Not Fraud'}**")
