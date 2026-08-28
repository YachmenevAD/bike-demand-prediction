import pandas as pd


DATA_PATH = "data/sample/bike_rentals.csv"
VALIDATION_START = pd.Timestamp("2024-01-02 00:00:00")


def calculate_mean_rentals(data: pd.DataFrame) -> float:
    return float(data["rentals"].mean())

def calculate_mae(
    actual: list[int],
    predicted: list[float],
) -> float:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted must have the same length")

    total_absolute_error = 0.0

    for actual_value, predicted_value in zip(actual, predicted):
        total_absolute_error += abs(actual_value - predicted_value)

    return total_absolute_error / len(actual)


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    data = data.sort_values("timestamp")

    train_mask = data["timestamp"] < VALIDATION_START
    validation_mask = data["timestamp"] >= VALIDATION_START

    train_data = data.loc[train_mask]
    validation_data = data.loc[validation_mask]

    if train_data.empty or validation_data.empty:
        raise ValueError("Train and validation datasets must not be empty")

    baseline_prediction = calculate_mean_rentals(train_data)

    actual = validation_data["rentals"].tolist()
    predicted = [
        baseline_prediction
        for _ in actual
    ]

    mae = calculate_mae(actual, predicted)

    print(f"Train rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Last train timestamp: {train_data['timestamp'].max()}")
    print(
        "First validation timestamp: "
        f"{validation_data['timestamp'].min()}"
    )
    print(f"Baseline prediction: {baseline_prediction:.1f}")
    print(f"Validation actual: {actual}")
    print(f"Validation predicted: {predicted}")
    print(f"Validation MAE: {mae:.2f}")


if __name__ == "__main__":
    main()