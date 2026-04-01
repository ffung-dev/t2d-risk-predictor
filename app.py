import streamlit as st
import pandas as pd
import joblib

# load models
lr_model = joblib.load("models/logreg_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")

# app 
st.title("Type 2 Diabetes Risk Predictor")
st.write("Enter patient information to predict diabetes risk.")

# get patient info
gender = st.radio("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
glucose = st.number_input("Blood Glucose Level", min_value=0.0)
hypertension = st.radio("Hypertension", ["Yes", "No"])
hba1c = st.number_input("HbA1c Level", min_value=0.0)

# gender and hypertension to num
gender_val = 1 if gender == "Female" else 0
hypertension_val = 1 if hypertension == "Yes" else 0

def assign_risk(prob):
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.7:
        return "High"
    else:
        return "Very High"
    
if st.button("Predict Risk"):
    # input DataFrame 
    input_data = pd.DataFrame([{
        'gender': gender_val,
        'age': age,
        'bmi': bmi,
        'blood_glucose_level': glucose,
        'hypertension': hypertension_val,
        'hbA1c_level': hba1c
    }])

    # Logistic Regression
    prob_log = lr_model.predict_proba(input_data)[:,1][0]
    risk_log = assign_risk(prob_log)

    # Random Forest
    prob_rf = rf_model.predict_proba(input_data)[:,1][0]
    risk_rf = assign_risk(prob_rf)

    # display results
    st.subheader("Results")

    st.write("Logistic Regression")
    st.write(f"Probability: {prob_log:.2f}")
    st.write(f"Risk Level: {risk_log}")

    st.write("Random Forest")
    st.write(f"Probability: {prob_rf:.2f}")
    st.write(f"Risk Level: {risk_rf}")