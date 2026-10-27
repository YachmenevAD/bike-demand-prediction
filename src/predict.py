from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.real_features import FEATURE_COLUMNS
from src.train import MODEL_PATH

SAMPLE_MORNING_RUSH = {
    "season": 4,
    "mnth": 10,
    "hr": 8,
    "weekday": 2,
    "weathersit": 1,
    "yr": 1,
    "holiday": 0,
    "workingday": 1,
    "temp": 0.54,
    "atemp": 0.5152,
    "hum": 0.68,
    "windspeed": 0.1642,
}

SAMPLE_NIGHT_RAIN = {
    "season": 4,
    "mnth": 10,
    "hr": 3,
    "weekday": 2,
    "weathersit": 3,
    "yr": 1,
    "holiday": 0,
    "workingday": 1,
    "temp": 0.42,
    "atemp": 0.4242,
    "hum": 0.88,
    "windspeed": 0.2985,
}


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run 'python -m src.train' first."
        )

    return joblib.load(model_path)


def predict_demand(
    model: Pipeline,
    features: dict[str, int | float],
) -> float:
    missing_features = sorted(set(FEATURE_COLUMNS) - set(features.keys()))
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    input_frame = pd.DataFrame([features])[FEATURE_COLUMNS]
    predictions = model.predict(input_frame)

    if isinstance(predictions, tuple):
        raise TypeError("Expected a single array of predictions")

    predicted_value = float(predictions[0])
    return max(0.0, predicted_value)


def main() -> None:
    model = load_model()

    morning_prediction = predict_demand(model, SAMPLE_MORNING_RUSH)
    night_prediction = predict_demand(model, SAMPLE_NIGHT_RAIN)

    print(
        "Morning rush hour (08:00, clear weather, workday): "
        f"{morning_prediction:.1f} bikes"
    )
    print(f"Night rain (03:00, light rain, workday): {night_prediction:.1f} bikes")


if __name__ == "__main__":
    main()
