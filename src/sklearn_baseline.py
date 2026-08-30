import pandas as pd
from sklearn.dummy import DummyRegressor
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

    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    mae = mean_absolute_error(y_validation, predictions)

    print(f"Train rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Predictions: {predictions}")
    print(f"Validation MAE: {mae:.2f}")


if __name__ == "__main__":
    main()