import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

st.title("🎓 Student Academic Risk Predictor")

st.write("Enter student details to predict Pass / Fail risk.")

# Inputs
studytime = st.selectbox("Study Time (1 = low, 4 = high)", [1,2,3,4])
failures = st.number_input("Past Failures", 0, 4, 0)
absences = st.number_input("Absences", 0, 100, 0)

schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])
paid = st.selectbox("Paid Classes", ["yes", "no"])
activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
internet = st.selectbox("Internet Access", ["yes", "no"])
romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

# Encode yes/no
def encode(x):
    return 1 if x == "yes" else 0

features = np.array([[
    studytime,
    failures,
    absences,
    encode(schoolsup),
    encode(famsup),
    encode(paid),
    encode(activities),
    encode(internet),
    encode(romantic)
]])

# Scale
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict"):
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    # Risk score (probability of failing)
    risk_score = 1 - prob

    # Risk level logic
    if risk_score < 0.4:
        risk_level = "Low"
        color = "green"
    elif risk_score < 0.7:
        risk_level = "Moderate"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    st.markdown(f"""### 🎯 Risk Assessment Result- **Risk Level:** <span style='color:{color}'>{risk_level}</span>- **Risk Score:** {risk_score*100:.1f}%
    📌 *This is an early-warning estimate based on behavioral and support-related factors only.*
    """, unsafe_allow_html=True)

