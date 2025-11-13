import joblib
import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

st.title("📈 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or stay using AI.")

# Load model, scaler, and feature names
model = tf.keras.models.load_model("models/churn_model.h5")
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_names = joblib.load("models/feature_names.pkl")

# Collect only some fields for demo
tenure = st.number_input("Tenure (months)", 0, 100, 10)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Example encoding
contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
payment_encoded = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method]

# Prepare DataFrame
input_df = pd.DataFrame([[tenure, monthly_charges, total_charges, contract_encoded, payment_encoded]], 
                        columns=['tenure','MonthlyCharges','TotalCharges','Contract','PaymentMethod'])

# Add missing columns (set to 0)
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_names]

# Scale
scaled_features = scaler.transform(input_df)

if st.button("🔍 Predict"):
    prediction = model.predict(scaled_features)
    if prediction[0] > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")