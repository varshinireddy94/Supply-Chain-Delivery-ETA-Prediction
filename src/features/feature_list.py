"""
Final feature set selected for the LightGBM ETA model, chosen from the 30+
engineered logistics features via SelectFromModel + SHAP-driven pruning
(see notebooks/2_feature_selection.ipynb in the original project).
"""

TARGET = "delivered_in_days"

SELECTED_FEATURES = [
    "distance_customer_seller_min",
    "product_height_cm_min",
    "distance_customer_seller_q25",
    "freight_value_min",
    "customer_lat",
    "order_approved_at_second",
    "customer_lng",
    "order_purchase_timestamp_month",
    "product_length_cm_min",
    "product_weight_g_min",
    "payment_value_q75",
    "payment_value_max",
    "payment_value_mean",
    "order_approved_at_month",
]

BASELINE_COLUMN = "estimated_delivery_in_days"  # current rule-based estimate
