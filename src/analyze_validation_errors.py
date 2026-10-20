import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data import load_hourly_data, split_by_time
from src.real_features import (
    TARGET_COLUMN,
    select_features_and_target,
)
from src.train_real_models import build_random_forest_pipeline


N_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 1
MAX_DEPTH = None


def summarize_errors(
    results: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    return (
        results.groupby(group_column, as_index=False)
        .agg(
            observations=("absolute_error", "size"),
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("absolute_error", "mean"),
        )
    )


def main() -> None:
    data = load_hourly_data()
    train_data, validation_data, _ = split_by_time(data)

    X_train, y_train = select_features_and_target(train_data)
    X_validation, y_validation = select_features_and_target(
        validation_data
    )

    model = build_random_forest_pipeline()
    model.set_params(
        model__n_estimators=N_ESTIMATORS,
        model__min_samples_leaf=MIN_SAMPLES_LEAF,
        model__max_depth=MAX_DEPTH,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    if isinstance(predictions, tuple):
        raise TypeError("Expected a single array of predictions")

    results = validation_data[
        [
            "timestamp",
            "mnth",
            "hr",
            "weekday",
            "workingday",
            "weathersit",
            TARGET_COLUMN,
        ]
    ].copy()

    results = results.rename(
        columns={TARGET_COLUMN: "actual"}
    )
    results["prediction"] = predictions
    results["error"] = results["prediction"] - results["actual"]
    results["absolute_error"] = results["error"].abs()

    validation_mae = mean_absolute_error(
        y_validation,
        predictions,
    )
    mean_signed_error = results["error"].mean()

    hour_summary = summarize_errors(results, "hr")
    weather_summary = summarize_errors(results, "weathersit")
    month_summary = summarize_errors(results, "mnth")

    hours_with_highest_error = (
        hour_summary
        .sort_values("mae", ascending=False)
        .head(5)
    )

    print(f"Validation rows: {len(results)}")
    print(f"Validation MAE: {validation_mae:.2f}")
    print(f"Mean signed error: {mean_signed_error:.2f}")

    print("\nFive hours with highest MAE:")
    print(
        hours_with_highest_error
        .round(2)
        .to_string(index=False)
    )

    print("\nErrors by weather category:")
    print(
        weather_summary
        .sort_values("weathersit")
        .round(2)
        .to_string(index=False)
    )

    print("\nErrors by month:")
    print(
        month_summary
        .sort_values("mnth")
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()