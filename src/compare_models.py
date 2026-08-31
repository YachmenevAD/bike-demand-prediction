import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_time_features,
)


DATA_PATH = "data/sample/bike_rentals.csv"
VALIDATION_START = pd.Timestamp("2024-01-02 00:00:00")


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    data = data.sort_values("timestamp")
    featured_data = add_time_features(data)

    train_data = featured_data.loc[
        featured_data["timestamp"] < VALIDATION_START
    ]
    validation_data = featured_data.loc[
        featured_data["timestamp"] >= VALIDATION_START
    ]

    if train_data.empty or validation_data.empty:
        raise ValueError("Train and validation datasets must not be empty")

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data[TARGET_COLUMN]

    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    dummy_predictions = dummy_model.predict(X_validation)
    dummy_mae = mean_absolute_error(
        y_validation,
        dummy_predictions,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_validation)
    linear_mae = mean_absolute_error(
        y_validation,
        linear_predictions,
    )

    formatted_dummy_predictions = [
        round(float(value), 2)
        for value in dummy_predictions
    ]
    formatted_linear_predictions = [
        round(float(value), 2)
        for value in linear_predictions
    ]

    print(f"Train rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")

    print("\nDummyRegressor:")
    print(f"Predictions: {formatted_dummy_predictions}")
    print(f"MAE: {dummy_mae:.2f}")

    print("\nLinearRegression:")
    print(f"Predictions: {formatted_linear_predictions}")
    print(f"MAE: {linear_mae:.2f}")

    print("\nLinear regression parameters:")
    print(f"Intercept: {linear_model.intercept_:.2f}")

    for feature_name, coefficient in zip(
        FEATURE_COLUMNS,
        linear_model.coef_,
    ):
        print(f"{feature_name}: {coefficient:.2f}")


if __name__ == "__main__":
    main()