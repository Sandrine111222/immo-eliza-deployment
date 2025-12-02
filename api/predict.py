# predict.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

_model = None


# 1. List of required features

MODEL_FEATURES = [
    'build_year', 'facades', 'garden', 'living_area', 'locality_name',
    'number_rooms', 'postal_code', 'property_id', 'property_type',
    'property_url', 'state', 'swimming_pool', 'terrace', 'province',
    'property_type_name', 'state_mapped', 'region', 'has_garden'
]


# 2. Load model

def load_model(path: str = r"C:\Users\sandy\immo-eliza-deployment\api\best_house_price_model.pkl"):
    global _model
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
        raise ValueError("JSON payload must be a dict or a list of dicts")

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


# 5. Manual test (optional)

if __name__ == "__main__":
    load_model()

    sample = {
        "number_rooms": 3,
        "living_area": 75,
        "locality_name": "Antwerpen"
        # Everything else will automatically be filled as None
    }

    print(predict_json(sample))