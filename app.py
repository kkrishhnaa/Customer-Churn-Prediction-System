import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Customer Churn Prediction System

**Algorithm:** XGBoost

**Dataset:** Telco Customer Churn

**Features:** 29

**Purpose:**
Predict whether a telecom customer is likely to leave the company.
""")

# ---------------- MAIN TITLE ----------------
st.title("📊 Customer Churn Prediction System")

st.markdown("""
Predict whether a telecom customer is likely to churn using Machine Learning.
""")

st.divider()

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Churn"):

    data = pd.DataFrame(
        [[0] * len(columns)],
        columns=columns
    )

    # Fill values
    data["gender"] = 1 if gender == "Male" else 0
    data["SeniorCitizen"] = 1 if senior == "Yes" else 0
    data["Partner"] = 1 if partner == "Yes" else 0
    data["Dependents"] = 1 if dependents == "Yes" else 0
    data["tenure"] = tenure
    data["PhoneService"] = 1 if phone == "Yes" else 0
    data["PaperlessBilling"] = 1 if paperless == "Yes" else 0
    data["MonthlyCharges"] = monthly

    # Scale data
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    st.divider()

    if prediction[0] == 1:

        st.error("⚠️ High Churn Risk")

        st.markdown("""
        ### Recommendation:
        - Offer retention discounts.
        - Provide better customer support.
        - Suggest long-term plans.
        - Contact the customer proactively.
        """)

    else:

        st.success("✅ Customer is likely to stay.")

        st.markdown("""
        ### Recommendation:
        - Maintain customer satisfaction.
        - Continue current services.
        - Offer loyalty rewards.
        """)

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Developed using Streamlit, XGBoost, Scikit-Learn and Python."
)