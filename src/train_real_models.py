from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.data import load_hourly_data, split_by_time
from src.real_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    select_features_and_target,
)


def build_linear_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> float:
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, predictions)

    print(f"{name} MAE: {mae:.2f}")
    return float(mae)


def main() -> None:
    data = load_hourly_data()
    train_data, validation_data, _ = split_by_time(data)

    X_train, y_train = select_features_and_target(train_data)
    X_validation, y_validation = select_features_and_target(
        validation_data
    )

    dummy_model = DummyRegressor(strategy="mean")
    linear_pipeline = build_linear_pipeline()
    random_forest_pipeline = build_random_forest_pipeline()

    dummy_mae = evaluate_model(
        "DummyRegressor",
        dummy_model,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    linear_mae = evaluate_model(
        "LinearRegression",
        linear_pipeline,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    random_forest_mae = evaluate_model(
        "RandomForestRegressor",
        random_forest_pipeline,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    improvement = dummy_mae - linear_mae
    improvement_percent = improvement / dummy_mae * 100

    print(
        "Linear regression improvement over baseline: "
        f"{improvement:.2f} MAE "
        f"({improvement_percent:.1f}%)"
    )


    random_forest_improvement = dummy_mae - random_forest_mae
    random_forest_improvement_percent = (
        random_forest_improvement / dummy_mae * 100
    )

    print(
        "Random forest improvement over baseline: "
        f"{random_forest_improvement:.2f} MAE "
        f"({random_forest_improvement_percent:.1f}%)"
    )


if __name__ == "__main__":
    main()