train_records = [
    {"timestamp": "2024-01-01 06:00", "temperature": 8.1, "rentals": 16},
    {"timestamp": "2024-01-01 07:00", "temperature": 8.7, "rentals": 20},
    {"timestamp": "2024-01-01 08:00", "temperature": 9.4, "rentals": 35},
    {"timestamp": "2024-01-01 17:00", "temperature": 13.2, "rentals": 42},
    {"timestamp": "2024-01-01 18:00", "temperature": 12.5, "rentals": 38},
    {"timestamp": "2024-01-01 19:00", "temperature": 11.7, "rentals": 29},
]

validation_records = [
    {"timestamp": "2024-01-02 07:00", "temperature": 9.0, "rentals": 24},
    {"timestamp": "2024-01-02 08:00", "temperature": 10.1, "rentals": 40},
    {"timestamp": "2024-01-02 18:00", "temperature": 12.0, "rentals": 31},
]


def calculate_mean_rentals(records: list[dict]) -> float:
    total = 0

    for record in records:
        total += record["rentals"]

    return total / len(records)

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
    baseline_prediction = calculate_mean_rentals(train_records)

    actual = [record["rentals"] for record in validation_records]
    predicted = [
        baseline_prediction
        for _ in validation_records
    ]

    mae = calculate_mae(actual, predicted)

    print(f"Baseline prediction: {baseline_prediction:.1f}")
    print(f"Validation actual: {actual}")
    print(f"Validation predicted: {predicted}")
    print(f"Validation MAE: {mae:.2f}")


if __name__ == "__main__":
    main()