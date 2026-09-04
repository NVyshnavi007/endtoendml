"""
FastAPI service for the California housing price model
(RandomForestRegressor, trained in eda.ipynb)

Run:
    uvicorn main:app --reload

Expects the trained model file in the same directory as this file:
    best_model.pkl        (pickle)  OR
    house_price_model.joblib (joblib)
Set MODEL_PATH below to whichever one you actually have.
"""

import joblib
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "house_price_model.joblib"

# Exact column order the model was trained on (X_train.columns in the notebook).
# longitude, latitude, ..., median_income, then the one-hot dummy columns
# pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True) produces
# (alphabetical, '<1H OCEAN' dropped as the reference category).
FEATURE_ORDER = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
]

OceanProximity = Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# --------------------------------------------------------------------------
# Load model at startup
# --------------------------------------------------------------------------

app = FastAPI(title="House Price Prediction API", version="1.0.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Copy best_model.pkl (or update MODEL_PATH) into this directory."
        )
    model = joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class HouseFeatures(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: OceanProximity = Field(..., example="NEAR BAY")


class PredictionResponse(BaseModel):
    predicted_median_house_value: float


# --------------------------------------------------------------------------
# Preprocessing — mirrors the notebook's pd.get_dummies(drop_first=True)
# without depending on which categories happen to appear in a given request
# --------------------------------------------------------------------------

def preprocess(features: HouseFeatures) -> pd.DataFrame:
    row = features.dict()
    ocean = row.pop("ocean_proximity")

    # one-hot manually against the fixed FEATURE_ORDER, '<1H OCEAN' is baseline (all zeros)
    for col in FEATURE_ORDER:
        if col.startswith("ocean_proximity_"):
            row[col] = 1 if col == f"ocean_proximity_{ocean}" else 0

    df = pd.DataFrame([row])
    return df[FEATURE_ORDER]  # enforce exact training column order


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = preprocess(features)
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictionResponse(predicted_median_house_value=float(pred))
