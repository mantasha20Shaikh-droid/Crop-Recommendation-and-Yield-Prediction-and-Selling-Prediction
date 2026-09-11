"""AgriSmart ML training pipeline for recommendation, yield and selling-price models."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# PROJECT PATHS
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EDA_DIR = BASE_DIR / "eda"
MODEL_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(exist_ok=True)
RECOMMENDATION_DATASET = BASE_DIR / "Crop_Recommendation_Final.csv"
YIELD_DATASET = BASE_DIR / "Crop_Yield_Final.csv"
SELLING_DATASET = BASE_DIR / "Crop_Selling_Final.csv"

# HELPER FUNCTIONS
def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str
) -> None:
    """Check whether all required columns exist in a dataset."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing the following columns: {missing}"
        )

def eda(df: pd.DataFrame, name: str) -> dict:
    """Perform basic EDA and save numeric histograms."""
    print(f"\nEDA: {name}")
    print("Shape:", df.shape)
    missing = df.isnull().sum()
    duplicates = int(df.duplicated().sum())
    print("Missing Values:")
    print(missing)
    print("Duplicate Rows:", duplicates)
    # Create histograms for numeric columns
    for column in df.select_dtypes(include=np.number).columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()
        if values.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=30)
        plt.title(f"{name} - {column}")
        plt.xlabel(str(column))
        plt.ylabel("Frequency")
        plt.tight_layout()
        safe_column = (
            str(column)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        output_file = EDA_DIR / f"{name}_{safe_column}.png"
        plt.savefig(
            output_file,
            dpi=120
        )
        plt.close()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(missing.sum()),
        "duplicate_rows": duplicates,
    }

# CROP RECOMMENDATION MODEL
def train_recommendation_model():
    """Train Random Forest Classifier for crop recommendation."""
    print("\n------------------------------------")
    print("TRAINING CROP RECOMMENDATION MODEL")
    print("------------------------------------")
    if not RECOMMENDATION_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RECOMMENDATION_DATASET}"
        )
    df = pd.read_csv(RECOMMENDATION_DATASET)
    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]
    target = "label"
    require_columns(
        df,
        features + [target],
        "Recommendation dataset"
    )
    df = df[features + [target]].dropna().copy()
    if df.empty:
        raise ValueError(
            "Recommendation dataset contains no valid rows after removing missing values."
        )
    eda_info = eda(
        df,
        "Crop_Recommendation"
    )
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(
        X_train,
        y_train
    )
    predictions = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    ) * 100
    model_path = MODEL_DIR / "rf_recommendation.joblib"
    joblib.dump(
        model,
        model_path
    )
    print(f"Model saved: {model_path}")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")
    print(f"F1 Score : {f1:.2f}%")
    return {
        "model": "Random Forest Classifier",
        "task": "Classification",
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "features": features,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        },
        "dataset": eda_info,
    }
# CROP YIELD MODEL
def train_yield_model():
    """Train Random Forest Regressor for crop yield prediction."""
    print("\n------------------------------------")
    print("TRAINING CROP YIELD MODEL")
    print("------------------------------------")
    if not YIELD_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{YIELD_DATASET}"
        )
    df = pd.read_csv(YIELD_DATASET)
    features = [
        "Crop",
        "Season",
        "State",
        "Area",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
    ]
    target = "Yield"
    require_columns(
        df,
        features + [target],
        "Yield dataset"
    )
    df = df[features + [target]].dropna().copy()
    if df.empty:
        raise ValueError(
            "Yield dataset contains no valid rows after removing missing values."
        )
    eda_info = eda(
        df,
        "Crop_Yield"
    )
    # Convert categorical columns into numerical dummy variables
    X = pd.get_dummies(
        df[features],
        columns=[
            "Crop",
            "Season",
            "State",
        ],
        drop_first=True,
    )
    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if X.empty:
        raise ValueError(
            "Yield dataset contains no valid numerical target values."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train
    )
    predictions = model.predict(X_test)
    r2 = r2_score(
        y_test,
        predictions
    ) * 100
    mae = mean_absolute_error(
        y_test,
        predictions
    )
    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    )
    model_path = MODEL_DIR / "rf_yield.joblib"
    features_path = MODEL_DIR / "yield_features.joblib"
    joblib.dump(
        model,
        model_path
    )
    joblib.dump(
        list(X.columns),
        features_path
    )
    print(f"Model saved: {model_path}")
    print(f"Features saved: {features_path}")
    print(f"R2 Score: {r2:.2f}%")
    print(f"MAE     : {mae:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    return {
        "model": "Random Forest Regressor",
        "task": "Regression",
        "r2": round(r2, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "features": features,
        "encoded_features": len(X.columns),
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        },
        "dataset": eda_info,
    }
# SELLING PRICE MODEL
def train_selling_model():
    """Train Random Forest Regressor for selling-price prediction."""
    print("\n------------------------------------")
    print("TRAINING SELLING PRICE MODEL")
    print("------------------------------------")
    if not SELLING_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{SELLING_DATASET}"
        )
    df = pd.read_csv(SELLING_DATASET)
    # Predict price per kg.
    # Total Value = predicted price × quantity.
    features = [
        "Crop",
        "State",
        "Season",
        "Quantity (kg)",
    ]
    target = "Price per kg (₹)"
    require_columns(
        df,
        features + [
            target,
            "Total Value (₹)",
        ],
        "Selling dataset"
    )
    df = df[
        features + [
            target,
            "Total Value (₹)",
        ]
    ].dropna().copy()
    if df.empty:
        raise ValueError(
            "Selling dataset contains no valid rows after removing missing values."
        )
    eda_info = eda(
        df,
        "Crop_Selling"
    )
    # Convert categorical columns into dummy variables
    X = pd.get_dummies(
        df[features],
        columns=[
            "Crop",
            "State",
            "Season",
        ],
        drop_first=True,
    )
    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if X.empty:
        raise ValueError(
            "Selling dataset contains no valid numerical target values."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train
    )
    predictions = model.predict(X_test)
    r2 = r2_score(
        y_test,
        predictions
    ) * 100
    mae = mean_absolute_error(
        y_test,
        predictions
    )
    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    )
    model_path = MODEL_DIR / "rf_selling.joblib"
    features_path = MODEL_DIR / "selling_features.joblib"
    joblib.dump(
        model,
        model_path
    )
    joblib.dump(
        list(X.columns),
        features_path
    )
    print(f"Model saved: {model_path}")
    print(f"Features saved: {features_path}")
    print(f"R2 Score: {r2:.2f}%")
    print(f"MAE     : {mae:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    return {
        "model": "Random Forest Regressor",
        "task": "Regression",
        "target": "Price per kg (₹)",
        "r2": round(r2, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "features": features,
        "encoded_features": len(X.columns),
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        },
        "dataset": eda_info,
    }

# SAVE ML METADATA
def save_ml_metadata(
    recommendation,
    yield_prediction,
    selling_prediction,
):
    """Save training information for use by the application."""
    overall = (
        recommendation["accuracy"]
        + yield_prediction["r2"]
        + selling_prediction["r2"]
    ) / 3
    metadata = {
        "project": (
            "Crop Recommendation + Yield Prediction "
            "+ Selling Price Prediction"
        ),
        "trained_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "models": {
            "recommendation": recommendation,
            "yield": yield_prediction,
            "selling": selling_prediction,
        },
        "overall_combined_score": round(
            overall,
            2
        ),
    }
    metadata_path = MODEL_DIR / "ml_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nML metadata saved: {metadata_path}")
    return metadata

# MAIN
if __name__ == "__main__":
    print("\n====================================")
    print("       AGRISMART ML TRAINING")
    print("====================================")
    print(f"\nProject folder:")
    print(BASE_DIR)
    print("\nChecking datasets...")
    datasets = [
        RECOMMENDATION_DATASET,
        YIELD_DATASET,
        SELLING_DATASET,
    ]
    for dataset in datasets:
        if not dataset.exists():
            raise FileNotFoundError(
                f"\nDataset not found:\n{dataset}\n\n"
                "Make sure the CSV file is in the same folder "
                "as train_model.py."
            )
        print(f"✓ Found: {dataset.name}")
    # Train all models
    recommendation = train_recommendation_model()
    yield_prediction = train_yield_model()
    selling_prediction = train_selling_model()

    # Save metadata
    metadata = save_ml_metadata(
        recommendation,
        yield_prediction,
        selling_prediction,
    )

    # Final result
    print("\n====================================")
    print("   ALL MODELS TRAINED SUCCESSFULLY")
    print("====================================")
    print(
        f"\nCrop Recommendation Accuracy : "
        f"{recommendation['accuracy']:.2f}%"
    )
    print(
        f"Yield Prediction R2 Score    : "
        f"{yield_prediction['r2']:.2f}%"
    )
    print(
        f"Selling Price R2 Score       : "
        f"{selling_prediction['r2']:.2f}%"
    )
    print(
        f"Overall Combined Score       : "
        f"{metadata['overall_combined_score']:.2f}%"
    )
    print("\nGenerated files:")
    print(
        f"✓ {MODEL_DIR / 'rf_recommendation.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'rf_yield.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'yield_features.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'rf_selling.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'selling_features.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'ml_metadata.json'}"
    )
    print(
        f"✓ EDA graphs → {EDA_DIR}"
    )
    print("\n====================================")
    print("           TRAINING DONE")
    print("====================================\n")
