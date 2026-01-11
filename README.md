# Early Risk Detection for Student Academic Performance
## 📌 Problem Statement

Academic failure is often identified too late, after final exams.
This project focuses on early detection of at-risk students using machine learning, enabling timely academic intervention.

The system estimates academic risk scores (Low / Medium / High) as an early-warning signal, rather than making final pass/fail decisions.

## 📊 Dataset

The dataset contains student-level information including:
Study habits
Attendance
Academic background
Previous assessment scores
Only pre-final / mid-semester features are considered to avoid data leakage.

## 🎯 Objective

Identify students at risk of poor performance early
Build an interpretable ML model
Analyze key factors contributing to academic risk

## 🧠 Approach

Exploratory Data Analysis (EDA)
Feature engineering
Risk category creation
Model training & evaluation
Explainability & bias analysis

## 📈 Models Used

Logistic Regression (baseline)
Random Forest
Gradient Boosting
Evaluation metrics prioritize Recall and F1-score to minimize false negatives (missing at-risk students).

## 🔍 Key Insights

Attendance and prior scores are the strongest predictors
Performance trends matter more than absolute scores
Models can show bias if not evaluated across subgroups

## 🤖 Modeling Summary

Framed student performance prediction as a **risk assessment problem**
Converted final grades into a binary outcome for model training
Used Logistic Regression with class balancing to prioritize recall
Deployed predictions as **probability-based risk scores** instead of hard labels
Interpreted model behavior using coefficients and feature importance


## 🧪 Prediction Philosophy

The system is designed as an **early-warning model**, not a definitive outcome predictor.
Predictions are intentionally conservative when limited academic signals are available,
favoring recall over precision to avoid missing at-risk students.

## ⚠️ Ethical Considerations

Predictions should assist educators, not penalize students
Model decisions must be interpretable
Bias across demographic groups is explicitly evaluated
Predictions are presented as risk indicators, not deterministic judgments, to prevent misuse.


## 🚀 Future Work

Real-time dashboard for educators
Integration with institutional systems
Personalized intervention recommendations
Multi-stage prediction models (early-semester vs mid-semester risk)

