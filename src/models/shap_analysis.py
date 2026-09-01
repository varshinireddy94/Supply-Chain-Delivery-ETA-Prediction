"""
SHAP-based explainability for the tuned LightGBM ETA model.

Computes SHAP values on the OOT set, saves:
  - reports/figures/shap_summary.png   (beeswarm plot)
  - reports/shap_feature_importance.json  (mean |SHAP| per feature)

Run:
    python -m src.models.shap_analysis
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.features.feature_list import SELECTED_FEATURES

MODEL_PATH = Path("models/lgbm_eta_model.joblib")
DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")


def main():
    model = joblib.load(MODEL_PATH)
    oot = pd.read_parquet(DATA_DIR / "df_oot.parquet")
    X_oot = oot[SELECTED_FEATURES]

    # sample for speed; SHAP's TreeExplainer is exact but scales with rows
    X_sample = X_oot.sample(n=min(4000, len(X_oot)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    mean_abs_shap = (
        pd.Series(abs(shap_values.values).mean(axis=0), index=SELECTED_FEATURES)
        .sort_values(ascending=False)
    )
    print("Top features by mean |SHAP|:")
    print(mean_abs_shap.round(3).to_string())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "shap_feature_importance.json", "w") as f:
        json.dump(mean_abs_shap.round(4).to_dict(), f, indent=2)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "figures" / "shap_summary.png", dpi=150)
    print(f"\nSaved shap_summary.png and shap_feature_importance.json under {REPORT_DIR}/")


if __name__ == "__main__":
    main()
