"""
Feature engineering pipeline for the Delivery ETA Prediction Engine.

Builds an order-level feature table from the raw Olist e-commerce tables
(orders, order_items, order_payments, products, sellers, customers,
geolocation). Produces 30+ logistics features per order: seller-customer
distance, freight/payment/price statistics, product dimensions, and
purchase-to-approval timing.

Input : CSVs from the "Brazilian E-Commerce Public Dataset by Olist"
        placed under data/raw/ (see README for the download link).
Output: data/interim/features.parquet -- one row per order, ready to be
        split into train / valid / OOT by src/models/train_model.py.

Run:
    python -m src.features.build_features
"""

from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

AGG_STATS = ["count", "min", "max", "mean", "median",
             lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
AGG_NAMES = ["count", "min", "max", "mean", "median", "q25", "q75"]


def _agg(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate a numeric column per order, adding a `range` (max-min) stat."""
    g = df.groupby(group_col)[value_col].agg(AGG_STATS)
    g.columns = [f"{value_col}_{name}" for name in AGG_NAMES]
    g[f"{value_col}_range"] = g[f"{value_col}_max"] - g[f"{value_col}_min"]
    return g


def load_raw_tables() -> dict:
    names = [
        "olist_orders_dataset",
        "olist_order_items_dataset",
        "olist_order_payments_dataset",
        "olist_products_dataset",
        "olist_sellers_dataset",
        "olist_customers_dataset",
        "olist_geolocation_dataset",
    ]
    return {n.split("_dataset")[0].replace("olist_", ""): pd.read_csv(RAW_DIR / f"{n}.csv")
            for n in names}


def build_geo_lookup(geolocation: pd.DataFrame) -> pd.DataFrame:
    """One lat/lng per zip-code prefix (median of the reported points)."""
    return (
        geolocation
        .groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
        .reset_index()
    )


def add_customer_seller_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Great-circle (geodesic) distance in km between seller and customer."""
    def _dist(row):
        if pd.isna(row.customer_lat) or pd.isna(row.seller_lat):
            return np.nan
        return geodesic((row.customer_lat, row.customer_lng),
                         (row.seller_lat, row.seller_lng)).km

    df["distance_customer_seller"] = df.apply(_dist, axis=1)
    return df


def build_feature_table() -> pd.DataFrame:
    tables = load_raw_tables()
    orders, items, payments = tables["orders"], tables["order_items"], tables["order_payments"]
    products, sellers, customers = tables["products"], tables["sellers"], tables["customers"]
    geo_lookup = build_geo_lookup(tables["geolocation"])

    for col in ["order_purchase_timestamp", "order_approved_at",
                "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col])

    # customer / seller lat-lng
    customers = customers.merge(
        geo_lookup, left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix", how="left"
    ).rename(columns={"lat": "customer_lat", "lng": "customer_lng"})
    sellers = sellers.merge(
        geo_lookup, left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix", how="left"
    ).rename(columns={"lat": "seller_lat", "lng": "seller_lng"})

    base = (
        orders
        .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_city",
                           "customer_state", "customer_lat", "customer_lng"]],
               on="customer_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(sellers[["seller_id", "seller_lat", "seller_lng"]], on="seller_id", how="left")
    )
    base = add_customer_seller_distance(base)

    order_level = base.drop_duplicates("order_id").set_index("order_id")[
        ["order_purchase_timestamp", "order_approved_at",
         "order_delivered_customer_date", "order_estimated_delivery_date",
         "customer_zip_code_prefix", "customer_city", "customer_state",
         "customer_lat", "customer_lng"]
    ]

    feature_blocks = [
        _agg(base, "order_id", "distance_customer_seller"),
        _agg(payments, "order_id", "payment_value"),
        _agg(base, "order_id", "price"),
        _agg(base, "order_id", "freight_value"),
        _agg(base, "order_id", "product_weight_g"),
        _agg(base, "order_id", "product_length_cm"),
        _agg(base, "order_id", "product_height_cm"),
        _agg(base, "order_id", "product_width_cm"),
    ]
    features = order_level.join(feature_blocks, how="left")

    # purchase -> approval timing
    delta = (features["order_approved_at"] - features["order_purchase_timestamp"])
    features["order_purchase_until_approved_in_seconds"] = delta.dt.total_seconds()
    features["order_purchase_timestamp_month"] = features["order_purchase_timestamp"].dt.month
    features["order_approved_at_second"] = features["order_approved_at"].dt.second

    # target + business baseline
    features["delivered_in_days"] = (
        features["order_delivered_customer_date"] - features["order_purchase_timestamp"]
    ).dt.days
    features["estimated_delivery_in_days"] = (
        features["order_estimated_delivery_date"] - features["order_purchase_timestamp"]
    ).dt.days

    features = features.dropna(subset=["delivered_in_days"])

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "features.parquet"
    features.reset_index().to_parquet(out_path, index=False)
    print(f"Wrote {features.shape[0]:,} orders x {features.shape[1]} columns -> {out_path}")
    return features


if __name__ == "__main__":
    build_feature_table()
