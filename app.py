import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("artifacts/tuned_model_pipeline.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction (USD)")
st.markdown("Predict your salary based on your role, experience, and more.")

# --- Input Form ---
with st.form("prediction_form"):
    work_year = st.selectbox("Work Year", [2020, 2021, 2022, 2023, 2024])
    
    experience_level = st.selectbox("Experience Level", [
        "EN",  # Entry-level
        "MI",  # Mid-level
        "SE",  # Senior
        "EX"   # Executive
    ])
    
    employment_type = st.selectbox("Employment Type", [
        "FT",  # Full-time
        "PT",  # Part-time
        "CT",  # Contract
        "FL"   # Freelance
    ])
    
    job_title = st.text_input("Job Title", value="Data Scientist")

    employee_residence = st.text_input("Employee Residence Country Code (e.g., IN, US, DE)", value="IN")

    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 0, step=25)

    company_location = st.text_input("Company Location Country Code (e.g., IN, US, DE)", value="US")

    company_size = st.selectbox("Company Size", [
        "S",  # Small
        "M",  # Medium
        "L"   # Large
    ])

    submit = st.form_submit_button("Predict Salary 💰")

# --- Prediction Logic ---
if submit:
    try:
        input_df = pd.DataFrame({
            "work_year": [work_year],
            "experience_level": [experience_level],
            "employment_type": [employment_type],
            "job_title": [job_title],
            "employee_residence": [employee_residence],
            "remote_ratio": [remote_ratio],
            "company_location": [company_location],
            "company_size": [company_size]
        })

        prediction = model.predict(input_df)[0]
        formatted_salary = f"${prediction:,.2f}"

        st.success(f"💸 Predicted Salary (USD): **{formatted_salary}**")

    except Exception as e:
        st.error(f"❌ Error occurred during prediction: {e}")
