"""
Trains the delivery-ETA regressor.

1. Benchmarks XGBoost, LightGBM and CatBoost with default parameters using
   3-fold cross-validation on the training set.
2. Tunes the best-performing family (LightGBM) with Optuna.
3. Refits on the full training set and reports generalization on the
   out-of-time (OOT) holdout.
4. Saves the final model to models/lgbm_eta_model.joblib.

Run:
    python -m src.models.train_model
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

from src.features.feature_list import BASELINE_COLUMN, SELECTED_FEATURES, TARGET

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
N_FOLDS = 3
RANDOM_STATE = 42


def load_data():
    train = pd.read_parquet(DATA_DIR / "df_train.parquet")
    oot = pd.read_parquet(DATA_DIR / "df_oot.parquet")
    X_train, y_train = train[SELECTED_FEATURES], train[TARGET]
    X_oot, y_oot = oot[SELECTED_FEATURES], oot[TARGET]
    return train, oot, X_train, y_train, X_oot, y_oot


def score(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_baseline(oot: pd.DataFrame) -> dict:
    """Current rule-based estimate vs actual delivery time."""
    return score(oot[TARGET], oot[BASELINE_COLUMN])


def benchmark_models(X_train, y_train) -> dict:
    """3-fold CV comparison of default-parameter XGBoost / LightGBM / CatBoost."""
    models = {
        "xgboost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "lightgbm": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "catboost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
    }
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        results[name] = {
            "cv_r2_mean": cv_res["test_r2"].mean(), "cv_r2_std": cv_res["test_r2"].std(),
            "cv_rmse_mean": -cv_res["test_neg_rmse"].mean(), "cv_rmse_std": cv_res["test_neg_rmse"].std(),
            "cv_mae_mean": -cv_res["test_neg_mae"].mean(), "cv_mae_std": cv_res["test_neg_mae"].std(),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"[{name:8s}] CV MAE = {results[name]['cv_mae_mean']:.3f} "
              f"+/- {results[name]['cv_mae_std']:.3f}  ({results[name]['fit_seconds']}s)")
    return results


def tune_lightgbm(X_train, y_train, n_trials: int = 20) -> optuna.Study:
    """Optuna search over LightGBM hyperparameters, scored by 2-fold CV MAE."""
    cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        cv_res = cross_validate(model, X_train, y_train, cv=cv,
                                 scoring="neg_mean_absolute_error", n_jobs=1)
        return -cv_res["test_score"].mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, oot, X_train, y_train, X_oot, y_oot = load_data()
    print(f"Train: {train.shape[0]:,} orders | OOT: {oot.shape[0]:,} orders | "
          f"{len(SELECTED_FEATURES)} features\n")

    baseline_scores = evaluate_baseline(oot)
    print(f"[baseline] rule-based OOT MAE = {baseline_scores['mae']:.3f}, "
          f"R2 = {baseline_scores['r2']:.3f}\n")

    print("Step 1/2 - benchmarking default-parameter models (3-fold CV)...")
    bench_results = benchmark_models(X_train, y_train)

    print("\nStep 2/2 - tuning LightGBM with Optuna...")
    study = tune_lightgbm(X_train, y_train, n_trials=20)
    best_params = {**study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
    print(f"Best CV MAE = {study.best_value:.3f}")
    print(f"Best params: {best_params}")

    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    oot_pred = final_model.predict(X_oot)
    oot_scores = score(y_oot, oot_pred)
    print(f"\nFinal LightGBM OOT scores: MAE = {oot_scores['mae']:.3f}, "
          f"RMSE = {oot_scores['rmse']:.3f}, R2 = {oot_scores['r2']:.3f}")

    model_path = MODEL_DIR / "lgbm_eta_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\nSaved model -> {model_path}")

    report = {
        "n_train": int(train.shape[0]),
        "n_oot": int(oot.shape[0]),
        "features": SELECTED_FEATURES,
        "baseline_oot": baseline_scores,
        "benchmark_cv": bench_results,
        "optuna_best_params": best_params,
        "optuna_best_cv_mae": study.best_value,
        "final_oot": oot_scores,
    }
    with open(REPORT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved report -> {REPORT_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
