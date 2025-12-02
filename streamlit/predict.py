# predict.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
from pathlib import Path

_model = None


# 1. List of required features

MODEL_FEATURES = [
    'build_year', 'facades', 'garden', 'living_area', 'locality_name',
    'number_rooms', 'postal_code', 'property_id', 'property_type',
    'property_url', 'state', 'swimming_pool', 'terrace', 'province',
    'property_type_name', 'state_mapped', 'region', 'has_garden'
]

# 2. Load model

def load_model(path: str = None):
    """
    Loads the model from the streamlit folder.
    Defaults to: streamlit/best_house_price_model.pkl
    """
    global _model

    if path is None:
        # Auto-detect model file relative to predict.py
        base = Path(__file__).resolve().parent
        path = base / "best_house_price_model.pkl"

    _model = joblib.load(path)

def _ensure_loaded():
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model().")

# 3. Input → DataFrame

def _json_to_dataframe(payload: Union[Dict, List[Dict]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("JSON payload must be a dict or list of dicts")

    # Add missing columns with None
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = None

    # Ensure correct order
    df = df[MODEL_FEATURES]

    return df


# 4. Predict

def _predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    _ensure_loaded()
    preds = _model.predict(df)
    return preds.ravel()

def predict_json(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    df = _json_to_dataframe(payload)
    preds = _predict_dataframe(df)
    preds_list = [float(p) for p in preds]
    return {"predictions": preds_list}

def predict_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_json(record)

# 5. Manual test

if __name__ == "__main__":
    load_model()

    sample = {
        "number_rooms": 3,
        "living_area": 75,
        "locality_name": "Antwerpen"
    }

    print(predict_json(sample))
