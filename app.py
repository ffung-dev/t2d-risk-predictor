import streamlit as st
import pandas as pd
import joblib

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

h2, h3 {
    font-weight: 600;
}

.stButton>button {
    background-color: #3468b0;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    border: none;
}

.stButton>button:hover {
    background-color: #083370;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
</style>
""", unsafe_allow_html=True)

# load models
lr_model = joblib.load("models/logreg_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")

# app 
st.title("Type 2 Diabetes Risk Predictor")
st.write("Enter patient information to predict diabetes risk.")

# get patient info
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, step=1.0)
with col2: 
    hypertension = st.radio("Hypertension", ["Yes", "No"])
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, step=1.0)
    hba1c = st.number_input("HbA1c Level (%)", min_value=0.0, step=0.01)

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

# stylize
def color_risk(risk):
    if risk == "Low":
        return "#317628"
    elif risk == "Medium":
        return "#CFBD18"
    elif risk == "High":
        return "#EE8F37"
    else:
        return "#72130D"
 
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

    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background-color:#F9FAFB;">
    <b>Logistic Regression</b><br>
    Probability: {prob_log:.2f}<br>
    <span style="color:{color_risk(risk_log)}; font-weight:600;">
    Risk Level: {risk_log}
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background-color:#F9FAFB; margin-top:10px;">
    <b>Random Forest</b><br>
    Probability: {prob_rf:.2f}<br>
    <span style="color:{color_risk(risk_rf)}; font-weight:600;">
    Risk Level: {risk_rf}
    </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<p style="font-size:13px; color:#6B7280; text-align:center">
Type 2 Diabetes Risk Predictor<br>
<a href="link here" target="_blank">View Full Research Paper</a><br>
<a href="https://github.com/ffung-dev/t2d-risk-predictor" target="_blank">GitHub Repository</a><br>
WUHC Healthcare Hackathon 2026
</p>
""", unsafe_allow_html=True)