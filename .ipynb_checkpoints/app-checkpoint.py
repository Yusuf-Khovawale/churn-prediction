import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor + Explainability", layout="centered")
st.title("📉 Customer Churn Predictor — with Explainability")

# -------------------------
# Paths (app in project root)
# -------------------------
MODEL_XGB_PATH = os.path.join("models", "xgb_churn.joblib")
MODEL_LOGIT_PATH = os.path.join("models", "logit_churn.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")
FEATURES_PATH = os.path.join("models", "feature_columns.json")
NUMCOLS_PATH = os.path.join("models", "numeric_columns.json")

# -------------------------
# Load artifacts
# -------------------------
model = None
model_type = None

if os.path.exists(MODEL_XGB_PATH):
    model = joblib.load(MODEL_XGB_PATH)
    model_type = "xgb"
    st.success("Loaded XGBoost model.")
elif os.path.exists(MODEL_LOGIT_PATH):
    model = joblib.load(MODEL_LOGIT_PATH)
    model_type = "logit"
    st.success("Loaded Logistic Regression model.")
else:
    st.error("No model found. Train & save a model in the notebook first: models/xgb_churn.joblib or models/logit_churn.joblib")
    st.stop()

scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

try:
    with open(FEATURES_PATH) as f:
        TRAIN_COLS = json.load(f)
    with open(NUMCOLS_PATH) as f:
        NUM_COLS = json.load(f)
except Exception:
    st.error("Missing feature_columns.json / numeric_columns.json. Re-run the notebook cell that saves them.")
    st.stop()

# -------------------------
# Input form (raw schema like Telco CSV)
# -------------------------
st.subheader("Enter customer details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
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

pay_method = st.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
total = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=float(monthly*tenure), step=1.0)

# Engineered feature from notebook
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

st.caption("The app will apply the same one-hot encoding + column alignment (and scaling if used) as training.")

# -------------------------
# Preprocess exactly like training
# -------------------------
def preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df_raw, drop_first=True)
    # add any missing columns that existed during training
    for c in TRAIN_COLS:
        if c not in X.columns:
            X[c] = 0
    # drop any extra columns not seen during training
    drop_extras = [c for c in X.columns if c not in TRAIN_COLS]
    if drop_extras:
        X = X.drop(columns=drop_extras)
    # order columns
    X = X[TRAIN_COLS]
    # scale numeric cols if scaler is present (safe for both models)
    if scaler is not None and len(NUM_COLS) > 0:
        cols_to_scale = [c for c in NUM_COLS if c in X.columns]
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])
    return X

X_one = preprocess_raw(raw_one)

# -------------------------
# Predict
# -------------------------
if st.button("🔮 Predict churn probability"):
    try:
        proba = float(model.predict_proba(X_one)[:, 1][0])
        st.metric("Estimated churn probability", f"{proba:.2f}")
        st.success("Prediction complete ✓")
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")

st.divider()

# -------------------------
# Explainability: SHAP (XGBoost path) or coefficient contributions (Logistic)
# -------------------------
st.subheader("🔍 Explainability")

if model_type == "xgb":
    try:
        import shap
        # For tree models: use TreeExplainer; cast to float to avoid object dtype issues
        X_float = X_one.astype("float64")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_float)

        st.write("**Why this prediction?** (Waterfall for this customer)")
        # Waterfall for single row
        # Build Explanation object for nicer waterfall rendering if needed
        try:
            ex = shap.Explanation(
                values=shap_values[0],
                base_values=getattr(explainer, "expected_value", 0.0),
                data=X_float.iloc[0, :].values,
                feature_names=list(X_float.columns),
            )
            fig = plt.figure()
            shap.plots.waterfall(ex, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception:
            fig = plt.figure()
            shap.waterfall_plot(shap_values[0], feature_names=list(X_float.columns), show=False)
            st.pyplot(fig, clear_figure=True)

        # Optional: global summary if user uploads a small CSV (raw schema)
        st.write("**Global view (optional)** — upload a small CSV of customers in the *raw* schema to see overall top features:")
        up = st.file_uploader("Upload sample customers CSV (same columns as the original Telco dataset).", type=["csv"])
        if up is not None:
            raw_batch = pd.read_csv(up)
            X_batch = preprocess_raw(raw_batch).astype("float64")
            shap_vals_batch = explainer.shap_values(X_batch)
            fig2 = plt.figure()
            shap.summary_plot(shap_vals_batch, X_batch, plot_type="bar", show=False)
            st.pyplot(fig2, clear_figure=True)
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

elif model_type == "logit":
    # Fast, robust explanation using linear model coefficients on the scaled feature vector
    try:
        coef = model.coef_.reshape(-1)
        contrib = pd.Series(coef * X_one.values.reshape(-1), index=X_one.columns).sort_values(key=np.abs, ascending=False)
        topk = st.slider("Show top-k contributing features", 5, min(20, len(contrib)), 10)
        st.write("**Top contributions (positive push ↑ towards churn, negative push ↓ towards retention)**")
        fig3, ax3 = plt.subplots()
        contrib.head(topk).iloc[::-1].plot(kind="barh", ax=ax3)
        ax3.set_xlabel("Contribution (coef × feature value)")
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        st.caption("For linear models this bar chart is equivalent to SHAP for standardized features.")
    except Exception as e:
        st.error(f"Coefficient-based explanation failed: {e}")

st.divider()
st.caption("Tip: Keep feature names in README with a short plain-English note (e.g., 'Month-to-month contracts and higher monthly charges increase churn risk; longer tenure reduces risk').")
