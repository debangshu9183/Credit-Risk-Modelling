import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Load Model and Encoders
# =============================
model = joblib.load("random_forest_model.pkl")

# Load encoders for categorical columns (you saved them earlier)
encoders = {}
for col in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]:
    try:
        encoders[col] = joblib.load(f"{col}_encoder.pkl")
    except:
        pass

target_encoder = joblib.load("target-encoder.pkl")

# =============================
# Streamlit App Layout
# =============================
st.set_page_config(page_title="Credit Risk Predictor - HSBC", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Assessment App")
st.write("This app predicts whether a loan applicant is **Good Risk** or **Bad Risk** based on financial data.")

st.divider()

# =============================
# Collect User Inputs
# =============================
st.subheader("🧾 Applicant Information")

age = st.slider("Age", 18, 75, 35)
sex = st.selectbox("Sex", encoders["Sex"].classes_)
job = st.selectbox("Job (0=Unskilled, 1=Skilled, 2=Highly Skilled, 3=Management)", [0, 1, 2, 3])
housing = st.selectbox("Housing", encoders["Housing"].classes_)
saving_acc = st.selectbox("Saving Account", encoders["Saving accounts"].classes_)
checking_acc = st.selectbox("Checking Account", encoders["Checking account"].classes_)
credit_amt = st.number_input("Credit Amount (€)", min_value=250, max_value=20000, value=5000)
duration = st.slider("Loan Duration (Months)", 4, 72, 24)
purpose = st.selectbox("Loan Purpose", encoders["Purpose"].classes_)

# =============================
# Create DataFrame for Model
# =============================
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_acc])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_acc])[0]],
    "Credit amount": [credit_amt],
    "Duration": [duration],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]]
})

# =============================
# Make Prediction
# =============================
if st.button("🔍 Predict Credit Risk"):
    prediction = model.predict(input_data)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    if result == "good":
        st.success("✅ This applicant is likely a **Good Credit Risk**.")
    else:
        st.error("⚠️ This applicant is likely a **Bad Credit Risk**.")

    # Display raw probability
    prob = model.predict_proba(input_data)[0][prediction]
    st.write(f"**Confidence Score:** {prob*100:.2f}%")

# =============================
# Footer
# =============================
st.divider()
st.caption("Developed by Debangshu Sadhukhan | MSc Data Science | HSBC Credit Risk Project")
