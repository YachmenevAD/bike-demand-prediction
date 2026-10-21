import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from src.api import app
from src.data import TEST_START, VALIDATION_START, split_by_time
from src.predict import SAMPLE_MORNING_RUSH, predict_demand
from src.real_features import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    select_features_and_target,
)
from src.train_real_models import build_random_forest_pipeline


def make_synthetic_dataset() -> pd.DataFrame:
    timestamps = [
        pd.Timestamp("2012-06-20 00:00:00"),
        pd.Timestamp("2012-06-25 00:00:00"),
        pd.Timestamp("2012-07-05 00:00:00"),
        pd.Timestamp("2012-10-05 00:00:00"),
    ]
    rows = []
    for idx, ts in enumerate(timestamps, start=1):
        rows.append(
            {
                "instant": idx,
                "dteday": ts.normalize(),
                "timestamp": ts,
                "season": 3,
                "yr": 1,
                "mnth": ts.month,
                "hr": (idx * 4) % 24,
                "holiday": 0,
                "weekday": ts.dayofweek,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.5,
                "atemp": 0.5,
                "hum": 0.6,
                "windspeed": 0.2,
                "casual": 10 * idx,
                "registered": 40 * idx,
                "cnt": 50 * idx,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def trained_mini_model() -> Pipeline:
    data = make_synthetic_dataset()
    X, y = select_features_and_target(data)
    model = build_random_forest_pipeline()
    model.set_params(model__n_estimators=5, model__min_samples_leaf=1)
    model.fit(X, y)
    return model


def test_split_by_time_respects_chronological_order() -> None:
    data = make_synthetic_dataset()
    train_data, validation_data, test_data = split_by_time(data)

    assert len(train_data) == 2
    assert len(validation_data) == 1
    assert len(test_data) == 1
    assert train_data["timestamp"].max() < validation_data["timestamp"].min()
    assert validation_data["timestamp"].max() < test_data["timestamp"].min()


def test_select_features_excludes_forbidden_columns() -> None:
    data = make_synthetic_dataset()
    X, y = select_features_and_target(data)

    assert list(X.columns) == FEATURE_COLUMNS
    assert set(X.columns).isdisjoint(set(FORBIDDEN_FEATURES))
    assert len(y) == len(data)


def test_predict_demand_returns_non_negative_float(
    trained_mini_model: Pipeline,
) -> None:
    prediction = predict_demand(trained_mini_model, SAMPLE_MORNING_RUSH)

    assert isinstance(prediction, float)
    assert prediction >= 0.0


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid_and_invalid_input(
    trained_mini_model: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.get_model", lambda: trained_mini_model)
    client = TestClient(app)

    valid_response = client.post("/predict", json=SAMPLE_MORNING_RUSH)
    assert valid_response.status_code == 200
    assert valid_response.json()["predicted_rentals"] >= 0.0

    invalid_payload = dict(SAMPLE_MORNING_RUSH)
    invalid_payload["hr"] = 99
    invalid_response = client.post("/predict", json=invalid_payload)
    assert invalid_response.status_code == 422