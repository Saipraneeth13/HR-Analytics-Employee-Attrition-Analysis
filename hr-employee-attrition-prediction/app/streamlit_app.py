"""Interactive Streamlit HR Attrition Risk Simulator & Analytics Dashboard."""

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import engineer_features

st.set_page_config(page_title="HR Attrition Risk Predictor", page_icon="📊", layout="wide")
st.title("📊 HR Employee Attrition Analytics & Risk Predictor")
st.markdown("Predict individual employee flight risk and explore key retention drivers.")

@st.cache_resource
def load_model():
    return joblib.load("models/attrition_model.joblib")

model = load_model()

st.sidebar.header("Employee Profile Input")
age = st.sidebar.slider("Age", 18, 60, 32)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 3500, step=250)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "Sales Representative", "Laboratory Technician", "Human Resources",
        "Sales Executive", "Research Scientist", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Research Director",
    ],
)
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Frequently", "Travel_Rarely", "Non-Travel"])
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 2)
years_since_promo = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
years_with_mgr = st.sidebar.slider("Years With Current Manager", 0, 17, 1)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 2)
env_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 2)
work_life_balance = st.sidebar.slider("Work-Life Balance (1-4)", 1, 4, 2)

sample = pd.DataFrame([{
    "Age": age,
    "BusinessTravel": business_travel,
    "DailyRate": 800,
    "Department": department,
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": env_satisfaction,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 1 if monthly_income < 4000 else 2,
    "JobRole": job_role,
    "JobSatisfaction": job_satisfaction,
    "MaritalStatus": "Single",
    "MonthlyIncome": monthly_income,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "OverTime": overtime,
    "PercentSalaryHike": 13,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 0,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": min(years_at_company, 2),
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrManager": years_with_mgr,
}])

engineered = engineer_features(sample)
prob = float(model.predict_proba(engineered)[0, 1])

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Attrition Probability", f"{prob * 100:.1f}%")
with col2:
    tier = "🔴 High Risk" if prob >= 0.65 else ("🟡 Medium Risk" if prob >= 0.35 else "🟢 Low Risk")
    st.metric("Risk Classification", tier)
