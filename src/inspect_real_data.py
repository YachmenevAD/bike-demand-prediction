from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/hour.csv")

EXPECTED_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]

TARGET_COLUMN = "cnt"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. "
            "Run python scripts/download_data.py first."
        )

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["dteday"],
    )

    actual_columns = list(data.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset columns do not match the expected schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual: {actual_columns}"
        )

    data["timestamp"] = data["dteday"] + pd.to_timedelta(
        data["hr"],
        unit="h",
    )

    total_missing_values = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    duplicate_timestamps = int(data["timestamp"].duplicated().sum())

    target_identity_violations = int(
        (
            data[TARGET_COLUMN]
            != data["casual"] + data["registered"]
        ).sum()
    )

    print(f"Raw dataset shape: ({len(data)}, {len(EXPECTED_COLUMNS)})")

    print("\nColumn types:")
    print(data[EXPECTED_COLUMNS].dtypes)

    print("\nTime range:")
    print(f"Start: {data['timestamp'].min()}")
    print(f"End: {data['timestamp'].max()}")
    print(
        "Chronological order: "
        f"{data['timestamp'].is_monotonic_increasing}"
    )

    print("\nData quality:")
    print(f"Missing values: {total_missing_values}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate timestamps: {duplicate_timestamps}")
    print(
        "Target identity violations: "
        f"{target_identity_violations}"
    )

    print("\nTarget statistics:")
    print(data[TARGET_COLUMN].describe().round(2))


if __name__ == "__main__":
    main()