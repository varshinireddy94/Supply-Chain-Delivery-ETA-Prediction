# Supply Chain Delivery ETA Prediction Engine

Predicts e-commerce delivery time (in days) from order, seller, product, payment
and geospatial data, using a LightGBM model tuned with Optuna and explained with
SHAP. Served via FastAPI and a Streamlit demo.

Built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(96K+ orders).

## What's in this project

**1. Feature engineering** — `src/features/build_features.py`
Merges the raw Olist tables (orders, order items, payments, products, sellers,
customers, geolocation) into one row per order with 30+ logistics features:
seller-customer geodesic distance, freight value, product dimensions/weight,
payment behaviour, and purchase-to-approval timing — each aggregated as
count / min / max / mean / median / q25 / q75 / range where an order has
multiple items or payments.

**2. Model benchmarking & tuning** — `src/models/train_model.py`
- Benchmarks XGBoost, LightGBM and CatBoost with default parameters using
  3-fold cross-validation.
- Tunes the best family (LightGBM) with Optuna (TPE sampler, MAE objective).
- Refits on the full training set and validates on an out-of-time (OOT)
  holdout to check temporal generalization.

**3. Explainability & deployment** — `src/models/shap_analysis.py`, `api/app.py`, `streamlit_app/app.py`
- SHAP `TreeExplainer` identifies the strongest ETA drivers.
- FastAPI exposes the trained model as a `/predict` endpoint with a Pydantic
  schema and Swagger docs.
- Streamlit gives a simple interactive form to try predictions live.

## Results (this run, on the included data)

| Model              | CV MAE (days)  | Fit time |
|---------------------|:--------------:|:--------:|
| XGBoost (default)   | 5.28 ± 0.04    | 2.0s     |
| LightGBM (default)  | 5.21 ± 0.04    | 1.9s     |
| CatBoost (default)  | 5.18 ± 0.04    | 21.9s    |
| **LightGBM (Optuna-tuned)** | **CV MAE 5.16** | — |

**On the OOT holdout (18,603 unseen orders):**

| Metric | Rule-based baseline | Tuned LightGBM |
|--------|:-------------------:|:--------------:|
| MAE    | 12.76 days           | **4.34 days**  |
| RMSE   | 15.47 days           | 5.74 days      |
| R²     | -6.10                | 0.02           |

That's a **~66% reduction in MAE** over the current rule-based delivery
estimate. Full numbers are saved in `reports/training_report.json`.

**Top SHAP drivers** (`reports/figures/shap_summary.png`, `reports/shap_feature_importance.json`):
seller-customer distance and freight value dominate, followed by customer
latitude/longitude and purchase/approval month — consistent with the
hypothesis that geography and logistics cost drive delay more than product
attributes.

## Project layout

```
eta-engine/
├── data/
│   ├── raw/                  # place the Olist CSVs here to run build_features.py
│   └── processed/            # df_train / df_valid / df_oot (already engineered, included)
├── src/
│   ├── features/
│   │   ├── build_features.py # raw tables -> engineered feature table
│   │   └── feature_list.py   # final selected feature set + target
│   └── models/
│       ├── train_model.py    # benchmark + Optuna tuning + OOT evaluation
│       └── shap_analysis.py  # SHAP explainability
├── api/
│   └── app.py                # FastAPI service
├── streamlit_app/
│   └── app.py                # Streamlit demo
├── models/
│   └── lgbm_eta_model.joblib # trained model (included, ready to serve)
├── reports/
│   ├── training_report.json
│   ├── shap_feature_importance.json
│   └── figures/shap_summary.png
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# (optional) retrain from scratch — data/processed/*.parquet is already included
python -m src.models.train_model
python -m src.models.shap_analysis

# serve the model
uvicorn api.app:app --reload --port 8000
# -> open http://127.0.0.1:8000/docs

# or launch the interactive demo
streamlit run streamlit_app/app.py
```

To regenerate `data/processed/` from scratch instead of using the included
files, download the raw Olist CSVs into `data/raw/` and run:

```bash
python -m src.features.build_features
```

## Notes on scope

This repo intentionally covers only the modeling + serving slice described
above (feature engineering → benchmarking/tuning → explainability/deployment).
It does not include the SQL database setup, exploratory data analysis
notebooks, or mRMR-based feature selection from the original, longer-running
version of this project — those are separate steps you can layer back in
later if you want the fuller data-science-project scope.
