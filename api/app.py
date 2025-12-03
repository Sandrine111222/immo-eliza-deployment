# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import RootModel
from typing import Optional, Literal, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# 1. MODEL FEATURES


MODEL_FEATURES = [
    'build_year', 'facades', 'garden', 'living_area', 'locality_name',
    'number_rooms', 'postal_code', 'property_id', 'property_type',
    'property_url', 'state', 'swimming_pool', 'terrace', 'province',
    'property_type_name', 'state_mapped', 'region', 'has_garden'
]

_model = None  # will be loaded in lifespan



# 2. Load the model


def load_model(path: str = None):
    """
    Loads the model from best_house_price_model.pkl.
    """
    global _model

    if path is None:
        base = Path(__file__).resolve().parent
        path = base / "best_house_price_model.pkl"

    _model = joblib.load(path)


def _ensure_loaded():
    if _model is None:
        raise RuntimeError("Model is not loaded. (Check lifespan loader)")


# 3. Input → DataFrame


def _json_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    # Add missing columns
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = None

    return df[MODEL_FEATURES]


# 4. Predict


def _predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    _ensure_loaded()
    preds = _model.predict(df)
    return preds.ravel()


def predict_record(record: Dict[str, Any]) -> Dict[str, Any]:
    df = _json_to_dataframe(record)
    preds = _predict_dataframe(df)
    return {"predictions": [float(p) for p in preds]}


# 5. FastAPI Models (Input / Output)


class PropertyData(BaseModel):
    LivingArea: int = Field(...)
    TypeOfProperty: Literal["apartment", "house", "land", "office", "garage"]
    Bedrooms: int = Field(...)
    PostalCode: int = Field(...)
    SurfaceOfGood: Optional[int] = None
    Garden: Optional[bool] = None
    GardenArea: Optional[int] = None
    SwimmingPool: Optional[bool] = None
    Furnished: Optional[bool] = None
    Openfire: Optional[bool] = None
    Terrace: Optional[bool] = None
    NumberOfFacades: Optional[int] = None
    ConstructionYear: Optional[int] = None
    StateOfBuilding: Optional[
        Literal["to be done up", "to restore", "to renovate"]
    ] = None
    Kitchen: Optional[
        Literal["not installed", "usa not installed", "installed"]
    ] = None


class PredictionInput(RootModel):
    root: PropertyData


class PredictionOutput(BaseModel):
    prediction: Optional[float]
    status_code: int


# 6. Lifespan (modern startup loader)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    print("Model loaded 🌟")
    yield


app = FastAPI(title="Immo-Eliza Prediction API", lifespan=lifespan)



# 7. Routes


@app.get("/", response_model=str)
async def root():
    return "alive"


@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    try:
        record = input_data.root.model_dump()

        # Convert external names → model feature names
        mapped = {
            "living_area": record["LivingArea"],
            "property_type": record["TypeOfProperty"],
            "number_rooms": record["Bedrooms"],
            "postal_code": record["PostalCode"],
            "build_year": record["ConstructionYear"],
            "facades": record["NumberOfFacades"],
            "garden": record["Garden"],
            "has_garden": record["Garden"],
            "terrace": record["Terrace"],
            "swimming_pool": record["SwimmingPool"],
            "property_url": None,
            "property_id": None,
            "locality_name": None,
            "province": None,
            "region": None,
            "state": record["StateOfBuilding"],
            "state_mapped": record["StateOfBuilding"],
            "property_type_name": record["TypeOfProperty"],
        }

        # Predict
        result = predict_record(mapped)
        pred_value = result["predictions"][0]

        return PredictionOutput(
            prediction=pred_value,
            status_code=200
        )

    except Exception as e:
        return PredictionOutput(
            prediction=None,
            status_code=500
        )

