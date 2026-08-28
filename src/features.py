import pandas as pd


DATA_PATH = "data/sample/bike_rentals.csv"

FEATURE_COLUMNS = [
    "temperature",
    "hour",
    "day_of_week",
    "is_weekend",
]

TARGET_COLUMN = "rentals"


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    featured_data = data.copy()

    featured_data["hour"] = featured_data["timestamp"].dt.hour
    featured_data["day_of_week"] = featured_data["timestamp"].dt.dayofweek
    featured_data["is_weekend"] = featured_data["day_of_week"] >= 5

    return featured_data


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    featured_data = add_time_features(data)

    X = featured_data[FEATURE_COLUMNS]
    y = featured_data[TARGET_COLUMN]

    print("Feature table:")
    print(X)

    print("\nTarget:")
    print(y)

    print("\nObject types:")
    print(f"X: {type(X).__name__}")
    print(f"y: {type(y).__name__}")

    print("\nShapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")


if __name__ == "__main__":
    main()