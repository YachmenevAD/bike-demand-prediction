import pandas as pd

from src.data import load_hourly_data, split_by_time


CATEGORICAL_FEATURES = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit",
]

BINARY_FEATURES = [
    "yr",
    "holiday",
    "workingday",
]

NUMERIC_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + NUMERIC_FEATURES
)

TARGET_COLUMN = "cnt"

FORBIDDEN_FEATURES = [
    "instant",
    "dteday",
    "timestamp",
    "casual",
    "registered",
    TARGET_COLUMN,
]


def select_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = sorted(
        set(required_columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    forbidden_columns_in_features = sorted(
        set(FEATURE_COLUMNS) & set(FORBIDDEN_FEATURES)
    )

    if forbidden_columns_in_features:
        raise ValueError(
            "Forbidden columns found in features: "
            f"{forbidden_columns_in_features}"
        )

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    return X, y


def main() -> None:
    data = load_hourly_data()
    train_data, validation_data, test_data = split_by_time(data)

    X_train, y_train = select_features_and_target(train_data)
    X_validation, y_validation = select_features_and_target(
        validation_data
    )
    X_test, _ = select_features_and_target(test_data)

    same_feature_order = (
        list(X_train.columns)
        == list(X_validation.columns)
        == list(X_test.columns)
    )

    print("Feature groups:")
    print(f"Categorical: {CATEGORICAL_FEATURES}")
    print(f"Binary: {BINARY_FEATURES}")
    print(f"Numeric: {NUMERIC_FEATURES}")

    print("\nShapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"y_validation: {y_validation.shape}")
    print(f"X_test: {X_test.shape}")

    print("\nChecks:")
    print(f"Same feature order: {same_feature_order}")
    print(
        "Forbidden features included: "
        f"{sorted(set(X_train.columns) & set(FORBIDDEN_FEATURES))}"
    )

    print("\nTarget means:")
    print(f"Train: {y_train.mean():.2f}")
    print(f"Validation: {y_validation.mean():.2f}")


if __name__ == "__main__":
    main()