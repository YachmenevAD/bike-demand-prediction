from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/hour.csv")

VALIDATION_START = pd.Timestamp("2012-07-01 00:00:00")
TEST_START = pd.Timestamp("2012-10-01 00:00:00")


def load_hourly_data(
    data_path: Path = DATA_PATH,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Run python scripts/download_data.py first."
        )

    data = pd.read_csv(
        data_path,
        parse_dates=["dteday"],
    )

    data["timestamp"] = data["dteday"] + pd.to_timedelta(
        data["hr"],
        unit="h",
    )

    return data.sort_values("timestamp").reset_index(drop=True)


def split_by_time(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = data["timestamp"] < VALIDATION_START

    validation_mask = (
        (data["timestamp"] >= VALIDATION_START)
        & (data["timestamp"] < TEST_START)
    )

    test_mask = data["timestamp"] >= TEST_START


    train_data = data.loc[train_mask].copy()
    validation_data = data.loc[validation_mask].copy()
    test_data = data.loc[test_mask].copy()

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("Train, validation, and test must not be empty")

    assigned_rows = (
        len(train_data)
        + len(validation_data)
        + len(test_data)
    )

    if assigned_rows != len(data):
        raise ValueError("Some rows were not assigned to a data split")

    if train_data["timestamp"].max() >= validation_data["timestamp"].min():
        raise ValueError("Train and validation periods overlap")

    if validation_data["timestamp"].max() >= test_data["timestamp"].min():
        raise ValueError("Validation and test periods overlap")

    return train_data, validation_data, test_data


def print_split(name: str, data: pd.DataFrame) -> None:
    print(
        f"{name}: {len(data)} rows | "
        f"{data['timestamp'].min()} -> "
        f"{data['timestamp'].max()}"
    )


def main() -> None:
    data = load_hourly_data()

    train_data, validation_data, test_data = split_by_time(data)

    print(f"Total rows: {len(data)}")
    print_split("Train", train_data)
    print_split("Validation", validation_data)
    print_split("Test", test_data)


if __name__ == "__main__":
    main()