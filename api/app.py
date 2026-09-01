"""
FastAPI service for the Delivery ETA Prediction Engine.

Run locally:
    uvicorn api.app:app --reload --port 8000

Then POST to /predict with a JSON body matching OrderFeatures, or open
/docs for the interactive Swagger UI.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_list import SELECTED_FEATURES

MODEL_PATH = Path("models/lgbm_eta_model.joblib")

app = FastAPI(
    title="Supply Chain Delivery ETA Prediction Engine",
    description="Predicts delivery time (in days) for an e-commerce order "
                "using a LightGBM model tuned with Optuna.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run `python -m src.models.train_model` first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class OrderFeatures(BaseModel):
    distance_customer_seller_min: float = Field(..., description="Min seller-customer distance (km)")
    product_height_cm_min: float
    distance_customer_seller_q25: float
    freight_value_min: float
    customer_lat: float
    order_approved_at_second: float
    customer_lng: float
    order_purchase_timestamp_month: int = Field(..., ge=1, le=12)
    product_length_cm_min: float
    product_weight_g_min: float
    payment_value_q75: float
    payment_value_max: float
    payment_value_mean: float
    order_approved_at_month: int = Field(..., ge=1, le=12)

    class Config:
        json_schema_extra = {
            "example": {
                "distance_customer_seller_min": 803.96,
                "product_height_cm_min": 20.0,
                "distance_customer_seller_q25": 803.96,
                "freight_value_min": 20.03,
                "customer_lat": -16.7171085172,
                "order_approved_at_second": 3.0,
                "customer_lng": -43.8071416039,
                "order_purchase_timestamp_month": 7,
                "product_length_cm_min": 16.0,
                "product_weight_g_min": 1050.0,
                "payment_value_q75": 139.88,
                "payment_value_max": 139.88,
                "payment_value_mean": 139.88,
                "order_approved_at_month": 7,
            }
        }


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
    predicted_delivery_days_rounded: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Delivery ETA Prediction Engine is running. See /docs."}


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {"status": "ok" if model_ready else "model_missing", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderFeatures):
    model = get_model()
    row = pd.DataFrame([order.model_dump()])[SELECTED_FEATURES]
    pred_days = float(model.predict(row)[0])
    return PredictionResponse(
        predicted_delivery_days=round(pred_days, 2),
        predicted_delivery_days_rounded=round(pred_days),
    )
