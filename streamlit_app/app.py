"""
Streamlit demo for the Delivery ETA Prediction Engine.

Run:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.feature_list import SELECTED_FEATURES  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lgbm_eta_model.joblib"

st.set_page_config(page_title="Delivery ETA Prediction Engine", page_icon="📦", layout="centered")
st.title("📦 Supply Chain Delivery ETA Prediction Engine")
st.caption("LightGBM model tuned with Optuna · trained on 96K+ Olist e-commerce orders")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.models.train_model` first.")
    st.stop()

model = load_model()

st.subheader("Order details")
col1, col2 = st.columns(2)

with col1:
    distance_min = st.number_input("Seller–customer distance, min (km)", 0.0, 4000.0, 500.0)
    distance_q25 = st.number_input("Seller–customer distance, 25th pct (km)", 0.0, 4000.0, 500.0)
    freight_value_min = st.number_input("Freight value, min (R$)", 0.0, 500.0, 25.0)
    customer_lat = st.number_input("Customer latitude", -35.0, 6.0, -23.55)
    customer_lng = st.number_input("Customer longitude", -75.0, -30.0, -46.63)
    purchase_month = st.slider("Purchase month", 1, 12, 7)
    approved_month = st.slider("Approval month", 1, 12, 7)
    approved_second = st.number_input("Approval second-of-minute", 0, 59, 3)

with col2:
    product_height = st.number_input("Product height, min (cm)", 0.0, 200.0, 20.0)
    product_length = st.number_input("Product length, min (cm)", 0.0, 200.0, 20.0)
    product_weight = st.number_input("Product weight, min (g)", 0.0, 50000.0, 1000.0)
    payment_mean = st.number_input("Payment value, mean (R$)", 0.0, 5000.0, 150.0)
    payment_q75 = st.number_input("Payment value, 75th pct (R$)", 0.0, 5000.0, 150.0)
    payment_max = st.number_input("Payment value, max (R$)", 0.0, 5000.0, 150.0)

if st.button("Predict delivery time", type="primary"):
    row = pd.DataFrame([{
        "distance_customer_seller_min": distance_min,
        "product_height_cm_min": product_height,
        "distance_customer_seller_q25": distance_q25,
        "freight_value_min": freight_value_min,
        "customer_lat": customer_lat,
        "order_approved_at_second": approved_second,
        "customer_lng": customer_lng,
        "order_purchase_timestamp_month": purchase_month,
        "product_length_cm_min": product_length,
        "product_weight_g_min": product_weight,
        "payment_value_q75": payment_q75,
        "payment_value_max": payment_max,
        "payment_value_mean": payment_mean,
        "order_approved_at_month": approved_month,
    }])[SELECTED_FEATURES]

    pred_days = float(model.predict(row)[0])
    st.metric("Predicted delivery time", f"{pred_days:.1f} days", help="Model MAE on out-of-time data: 4.34 days")
    st.info(f"Suggested customer-facing window: **{max(round(pred_days) - 4, 0)}–{round(pred_days) + 4} days**")

st.divider()
st.caption(
    "Model: LightGBM (Optuna-tuned) · Benchmarked against XGBoost & CatBoost · "
    "SHAP shows seller–customer distance and freight value as the top ETA drivers."
)
