import pandas as pd


DATA_PATH = "data/sample/bike_rentals.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    print("First five rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumn types before conversion:")
    print(data.dtypes)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    print("\nColumn types after conversion:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isna().sum())

    minimum_rentals = data["rentals"].min()
    maximum_rentals = data["rentals"].max()

    print(f"\nMinimum rentals: {minimum_rentals}")
    print(f"Maximum rentals: {maximum_rentals}")


if __name__ == "__main__":
    main()