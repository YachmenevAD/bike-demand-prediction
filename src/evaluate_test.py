from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.data import load_hourly_data, split_by_time
from src.real_features import select_features_and_target
from src.train_real_models import (
    build_linear_pipeline,
    build_random_forest_pipeline,
)


N_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 1
MAX_DEPTH = None


def evaluate_on_test(
    name: str,
    model: Any,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, str | float]:
    model.fit(X_development, y_development)
    predictions = model.predict(X_test)

    if isinstance(predictions, tuple):
        raise TypeError("Expected a single array of predictions")

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def main() -> None:
    data = load_hourly_data()
    train_data, validation_data, test_data = split_by_time(data)

    development_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    X_development, y_development = select_features_and_target(
        development_data
    )
    X_test, y_test = select_features_and_target(test_data)

    tuned_random_forest = build_random_forest_pipeline()
    tuned_random_forest.set_params(
        model__n_estimators=N_ESTIMATORS,
        model__min_samples_leaf=MIN_SAMPLES_LEAF,
        model__max_depth=MAX_DEPTH,
    )

    models: list[tuple[str, Any]] = [
        ("DummyRegressor", DummyRegressor(strategy="mean")),
        ("LinearRegression", build_linear_pipeline()),
        ("RandomForestRegressor", tuned_random_forest),
    ]

    metrics_rows: list[dict[str, Any]] = []

    for model_name, model in models:
        metrics = evaluate_on_test(
            model_name,
            model,
            X_development,
            y_development,
            X_test,
            y_test,
        )
        metrics_rows.append(metrics)

    results_table = pd.DataFrame(metrics_rows)

    print(f"Development rows (train + validation): {len(X_development)}")
    print(f"Test rows: {len(X_test)}")
    print(
        f"Test period: {test_data['timestamp'].min()} -> "
        f"{test_data['timestamp'].max()}"
    )
    print("\nFinal test results:")
    print(results_table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()