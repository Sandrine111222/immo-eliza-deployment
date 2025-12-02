# test.py
import os
import traceback
import joblib
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost only
import xgboost as xgb


# LOAD DATA
def load_data(file_path, target="price"):
    df = pd.read_csv(file_path)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {file_path}")
    df = df.dropna(subset=[target])
    if "number_rooms" in df.columns:
        df["number_rooms"] = df["number_rooms"].clip(upper=15)
    else:
        raise KeyError("Column 'number_rooms' was not found in the dataset.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, df


# PREPROCESSING PIPELINE
def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features


# CREATE SEARCH OBJECT (RandomizedSearch only for XGB)
def build_search(model, param_grid, preprocessor, cv_strategy):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model)
    ])

    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        cv=cv_strategy,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=1,
        random_state=42
    )


# TUNE MODEL
def tune_model(model, param_grid, preprocessor, cv_strategy, X_train, y_train):
    try:
        search = build_search(model, param_grid, preprocessor, cv_strategy)
        print(f"[XGBoost] Starting tuning…")
        search.fit(X_train, y_train)
        print(f"[XGBoost] Done — best score: {search.best_score_}")
        return search.best_estimator_, search.best_params_
    except Exception:
        print("Error while tuning XGBoost:")
        traceback.print_exc()
        raise


# MODEL EVALUATION
def evaluate(model, X_train, y_train, X_test, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    mae_test  = float(mean_absolute_error(y_test, test_preds))
    r2_train  = float(r2_score(y_train, train_preds))
    r2_test   = float(r2_score(y_test, test_preds))

    print(f"\n===== XGBoost =====")
    print(f"RMSE (test): {rmse_test:.3f}")
    print(f"MAE  (test): {mae_test:.3f}")
    print(f"R²   (train): {r2_train:.3f}")
    print(f"R²   (test):  {r2_test:.3f}")

    return {
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_train": r2_train,
        "r2_test": r2_test
    }


# MAIN
if __name__ == "__main__":

    file_path = r"C:\Users\sandy\Desktop\cleaned_output.csv"
    X, y, df = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor, num_ftrs, cat_ftrs = make_preprocessor(X)

    # --- Only XGBoost ---
    model = xgb.XGBRegressor(
        random_state=42,
        verbosity=0,
        n_jobs=1
    )

    param_grid = {
        "regressor__n_estimators": [200, 300],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5, 7]
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n===== Hyperparameter Search (XGBoost Only) =====")

    best_estimator, best_params = tune_model(
        model, param_grid, preprocessor, cv_strategy, X_train, y_train
    )

    print("\n===== Evaluating Best XGBoost Model =====")
    evaluation_results = {
        "XGBoost": {
            "metrics": evaluate(best_estimator, X_train, y_train, X_test, y_test),
            "best_params": best_params
        }
    }

    joblib.dump(evaluation_results, "evaluation_results.pkl")
    print("\nSaved evaluation_results.pkl")

    joblib.dump(best_estimator, "best_house_price_model.pkl")
    print("Saved best_house_price_model.pkl")

    joblib.dump({"XGBoost": best_estimator}, "all_best_estimators.pkl")
    joblib.dump({"XGBoost": best_params}, "all_best_params.pkl")
    print("\nSaved all_best_estimators.pkl and all_best_params.pkl")

    print("\nDONE — XGBoost tuned, evaluated, and saved.")
