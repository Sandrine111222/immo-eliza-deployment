# app.py
from typing import Any, Dict
import os
import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Property price prediction API")

logger = logging.getLogger("uvicorn.error")


class PredictionRequest(BaseModel):
    # We accept free-form JSON. Use dict to allow arbitrary features.
    __root__: Dict[str, Any]


def load_model_and_preprocessor():
    """
    Try to load a pre-trained model and optional preprocessor from disk.
    Expected files:
      - model.pkl   (scikit-learn-like estimator with predict())
      - preprocessor.pkl (optional) - transformer that accepts DataFrame and returns transformed array
    If not found, returns a fallback dummy model.
    """
    model_path = "model.pkl"
    preproc_path = "preprocessor.pkl"
    model = None
    preprocessor = None

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Loaded model from %s", model_path)
        if os.path.exists(preproc_path):
            preprocessor = joblib.load(preproc_path)
            logger.info("Loaded preprocessor from %s", preproc_path)
    except Exception as e:
        logger.exception("Failed to load model/preprocessor: %s", e)

    # If no model found, create a simple fallback
    if model is None:
        class DummyModel:
            def predict(self, X):
                # if X is pandas DataFrame: compute simple score:
                if isinstance(X, pd.DataFrame):
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # simple heuristic: sum numeric features (scaled)
                        return (X[numeric_cols].sum(axis=1).values * 1000).astype(float)
                    else:
                        # no numeric data -> return constant
                        return np.array([100000.0] * len(X))
                # if numpy array:
                try:
                    return (np.sum(X, axis=1) * 1000).astype(float)
                except Exception:
                    return np.array([100000.0] * X.shape[0])

        model = DummyModel()
        logger.info("Using DummyModel fallback")

    return model, preprocessor


MODEL, PREPROCESSOR = load_model_and_preprocessor()


def preprocess_input(payload: Dict[str, Any]):
    """
    Convert incoming JSON (dict) into a DataFrame and apply any preprocessor if available.
    This function attempts to:
      - Convert top-level scalars into a single-row DataFrame
      - Flatten nested dicts one level (common for e.g. {"location": {"lat":..., "lng":...}})
      - Convert lists of numbers to repeated columns (list -> col_0, col_1, ...), but normally
        you should send all features as top-level scalar/strings.
    """
    try:
        # If payload already contains a top-level key whose value is a list of dicts (batch):
        # e.g. {"__root__": [{"a":1}, {"a":2}]} -- but Pydantic root wraps it; here we assume single property.
        # We'll just support single-instance payloads; user can post a list to batch later if desired.
        # Flatten one-level nested dicts:
        flat = {}
        for k, v in payload.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat[f"{k}_{subk}"] = subv
            elif isinstance(v, list):
                # if list of numbers -> expand
                if all(isinstance(x, (int, float)) for x in v):
                    for i, item in enumerate(v):
                        flat[f"{k}_{i}"] = item
                else:
                    # otherwise store as string
                    flat[k] = str(v)
            else:
                flat[k] = v

        df = pd.DataFrame([flat])

        # Optional preprocessing step (if preprocessor exists)
        if PREPROCESSOR is not None:
            # Expect PREPROCESSOR to accept a DataFrame and return a numpy array
            X = PREPROCESSOR.transform(df)
            return X, df
        else:
            # If no preprocessor, attempt to:
            # - Fill missing numeric with 0, convert non-numeric to category codes,
            # - Return numpy array of numeric columns + categorical codes
            df_filled = df.copy()
            for col in df_filled.columns:
                if pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].fillna(0)
                else:
                    # convert to category codes
                    df_filled[col] = df_filled[col].astype("category").cat.codes.replace(-1, 0)

            X = df_filled.values.astype(float)
            return X, df

    except Exception as e:
        logger.error("Preprocessing failed: %s\n%s", e, traceback.format_exc())
        raise


@app.get("/", response_model=str)
async def root():
    """
    Healthcheck endpoint.
    """
    return "alive"


@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Expect JSON body with property features. Example:
    {
      "area": 85,
      "bedrooms": 2,
      "bathrooms": 1,
      "location": {"lat": 50.85, "lng": 4.35},
      "has_garden": true,
      "year_built": 1995
    }
    Returns JSON:
    { "prediction": 123456.78, "model": "loaded" }
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    # Allow either {"__root__": {...}} from Pydantic root models OR raw dict
    if "__root__" in payload and isinstance(payload["__root__"], dict):
        payload = payload["__root__"]

    # Validate that we have a dict
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object with property features.")

    try:
        X, df = preprocess_input(payload)

        # prediction
        preds = MODEL.predict(X)
        # If model returns array-like
        try:
            pred_value = float(np.asarray(preds).ravel()[0])
        except Exception:
            # last resort: str
            pred_value = preds[0]

        # Return contextual info: the raw features (dataframe) and the prediction
        return {
            "prediction": pred_value,
            "model": "loaded" if os.path.exists("model.pkl") else "dummy",
            "input_preview": df.to_dict(orient="records")[0],
        }

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
