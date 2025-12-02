# train_dummy_model.py
# Run this locally if you want to create a sample scikit-learn model and preprocessor.
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# create synthetic dataset
df = pd.DataFrame({
    "area": np.random.randint(30, 200, size=500),
    "bedrooms": np.random.randint(1, 5, size=500),
    "bathrooms": np.random.randint(1, 3, size=500),
    "has_garden": np.random.choice([0, 1], size=500),
    "location_city": np.random.choice(["A", "B", "C"], size=500),
})
# synthetic target (not meaningful, just for demo)
y = df["area"] * 2000 + df["bedrooms"] * 10000 + df["has_garden"] * 5000 + np.random.normal(0, 20000, size=500)

numeric_features = ["area", "bedrooms", "bathrooms", "has_garden"]
categorical_features = ["location_city"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])

pipeline.fit(df, y)

# save whole pipeline as model.pkl (app expects model.pkl but also supports separate preprocessor)
joblib.dump(pipeline, "model.pkl")
print("Saved model.pkl (pipeline)")

# Also save preprocessor separately to demonstrate separate load path (optional)
joblib.dump(preprocessor, "preprocessor.pkl")
print("Saved preprocessor.pkl")
