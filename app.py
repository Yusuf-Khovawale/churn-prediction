import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Predictor (Pipeline)")

MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Train & save the pipeline from the notebook.")
    st.stop()

pipe = joblib.load(MODEL_PATH)
st.success("Loaded trained pipeline ✅")

# --- raw schema inputs (match dataset columns) ---
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("PhoneService", ["No", "Yes"])
    multiple = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
with col2:
    online_sec = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
    online_bak = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])
    device_prot = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
    stream_tv = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
    stream_mv = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["No", "Yes"])
pay_method = st.selectbox("PaymentMethod",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)
monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
total = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=float(monthly*tenure), step=1.0)

avg_spend = total / (tenure if tenure != 0 else 1)

raw_one = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_bak,
    "DeviceProtection": device_prot,
    "TechSupport": tech_support,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_mv,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": pay_method,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "AvgMonthlySpend": avg_spend
}])

if st.button("🔮 Predict churn probability"):
    try:
        proba = float(pipe.predict_proba(raw_one)[:, 1][0])
        st.metric("Estimated churn probability", f"{proba:.2f}")
        st.caption("Higher probability = higher churn risk.")
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")
