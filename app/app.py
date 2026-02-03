import joblib
import numpy as np
import base64
import streamlit as st

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("app/assets/bg1.jpg")
st.markdown("""
<style>
.main-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Academic Risk Assessment System", layout="centered")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}

.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle {
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 25px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #111827;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f1f5f9;
    border-left: 6px solid #2563eb;
    margin-top: 20px;
}

.low {
    color: #16a34a;
    font-weight: bold;
}

.moderate {
    color: #d97706;
    font-weight: bold;
}

.high {
    color: #dc2626;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load models and scalers
# -----------------------------
model_early = joblib.load("../model_early.pkl")
scaler_early = joblib.load("../scaler_early.pkl")

model_mid = joblib.load("../model_mid.pkl")
scaler_mid = joblib.load("../scaler_mid.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">Academic Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">An early-warning tool to estimate student academic risk based on behavioral and academic indicators.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Prediction Stage
# -----------------------------
st.markdown('<div class="section-title">Prediction Stage</div>', unsafe_allow_html=True)

stage = st.radio(
    "",
    [
        "Early Semester (behavioral factors only)",
        "Mid Semester (includes internal assessment)"
    ]
)

# -----------------------------
# Inputs
# -----------------------------
st.markdown('<div class="section-title">Student Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox("Study Time (1 = very low, 4 = very high)", [1, 2, 3, 4])
    failures = st.number_input("Number of Past Failures", 0, 4, 0)
    absences = st.number_input("Number of Absences", 0, 100, 0)
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])

with col2:
    paid = st.selectbox("Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

if stage == "Mid Semester (includes internal assessment)":
    G1 = st.number_input("Mid-Semester Score (G1)", 0, 20, 10)

# -----------------------------
# Helper function
# -----------------------------
def encode(val):
    return 1 if val == "yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Assess Academic Risk"):
    base_features = [
        studytime,
        failures,
        absences,
        encode(schoolsup),
        encode(famsup),
        encode(paid),
        encode(activities),
        encode(internet),
        encode(romantic)
    ]

    if stage == "Early Semester (behavioral factors only)":
        features = np.array([base_features])
        features_scaled = scaler_early.transform(features)
        model_used = model_early
    else:
        features = np.array([base_features + [G1]])
        features_scaled = scaler_mid.transform(features)
        model_used = model_mid

    prob_pass = model_used.predict_proba(features_scaled)[0][1]
    risk_score = 1 - prob_pass

    if risk_score < 0.4:
        risk_level = "Low"
        css_class = "low"
    elif risk_score < 0.7:
        risk_level = "Moderate"
        css_class = "moderate"
    else:
        risk_level = "High"
        css_class = "high"

    # -----------------------------
    # Output
    # -----------------------------
    st.markdown(f"""
    <div class="result-box">
        <h3>Risk Assessment Result</h3>
        <p>Risk Level: <span class="{css_class}">{risk_level}</span></p>
        <p>Risk Score: {risk_score*100:.1f}%</p>
        <p><i>This is a {stage.lower()} prediction based on available information.</i></p>
    </div>
    """, unsafe_allow_html=True)
