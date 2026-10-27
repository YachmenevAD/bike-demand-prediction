import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from src.data import load_hourly_data, split_by_time
from src.real_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    select_features_and_target,
)
from src.train_real_models import build_random_forest_pipeline

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

N_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 1
MAX_DEPTH = None


def build_final_model() -> Pipeline:
    model = build_random_forest_pipeline()
    model.set_params(
        model__n_estimators=N_ESTIMATORS,
        model__min_samples_leaf=MIN_SAMPLES_LEAF,
        model__max_depth=MAX_DEPTH,
    )
    return model


def main() -> None:
    data = load_hourly_data()
    train_data, validation_data, test_data = split_by_time(data)

    development_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    X_development, y_development = select_features_and_target(development_data)
    X_test, y_test = select_features_and_target(test_data)

    model = build_final_model()
    model.fit(X_development, y_development)

    predictions = model.predict(X_test)
    if isinstance(predictions, tuple):
        raise TypeError("Expected a single array of predictions")

    metrics = {
        "model_type": "RandomForestRegressor",
        "parameters": {
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_depth": MAX_DEPTH,
        },
        "data_rows": {
            "development": len(X_development),
            "test": len(X_test),
        },
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "test_metrics": {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "rmse": round(float(root_mean_squared_error(y_test, predictions)), 2),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Test metrics: {metrics['test_metrics']}")


if __name__ == "__main__":
    main()
