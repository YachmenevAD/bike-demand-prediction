from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from src.predict import SAMPLE_MORNING_RUSH, load_model, predict_demand


app = FastAPI(
    title="Bike Demand Prediction API",
    description="REST API for predicting hourly bike rental demand.",
    version="1.0.0",
)


class BikeDemandRequest(BaseModel):
    season: int = Field(..., ge=1, le=4, description="Season (1: spring, 2: summer, 3: fall, 4: winter)")
    mnth: int = Field(..., ge=1, le=12, description="Month (1 to 12)")
    hr: int = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0 to 6)")
    weathersit: int = Field(..., ge=1, le=4, description="Weather category (1 to 4)")
    yr: int = Field(..., ge=0, le=1, description="Year index (0: 2011, 1: 2012)")
    holiday: int = Field(..., ge=0, le=1, description="1 if holiday, else 0")
    workingday: int = Field(..., ge=0, le=1, description="1 if working day, else 0")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    atemp: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized feeling temperature"
    )
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed")

    model_config = {
        "json_schema_extra": {
            "examples": [SAMPLE_MORNING_RUSH],
        }
    }


class BikeDemandResponse(BaseModel):
    predicted_rentals: float


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    return load_model()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=BikeDemandResponse)
def predict(request: BikeDemandRequest) -> BikeDemandResponse:
    model = get_model()
    features = request.model_dump()
    prediction = predict_demand(model, features)

    return BikeDemandResponse(predicted_rentals=round(prediction, 2))