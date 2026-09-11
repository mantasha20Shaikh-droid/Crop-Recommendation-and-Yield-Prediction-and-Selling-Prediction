"""AgriSmart FastAPI backend."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from database import (
    execute_admin_sql,
    get_all_history,
    get_history_by_id,
    get_ml_information,
    init_db,
    log_history,
    remove_history,
    update_history_record,
    verify_user,
)

# PROJECT PATHS AND SETTINGS
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ADMIN_API_KEY = os.getenv("AGRISMART_ADMIN_API_KEY", "")

# FASTAPI APPLICATION
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database when FastAPI starts."""
    init_db()
    yield

app = FastAPI(
    title="Crop Analytics FastAPI Backend",
    description=(
        "REST API for ML predictions, model information "
        "and history CRUD"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL LOADING
def load_model(filename: str):
    """Load a trained model from the models directory."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Model file not found: {path}. "
            "Run train_model.py first."
        )
    return joblib.load(path)

# Load models safely so that FastAPI can still start
# before the ML models have been trained.
try:
    rec_model = load_model("rf_recommendation.joblib")
    yield_model = load_model("rf_yield.joblib")
    yield_cols = load_model("yield_features.joblib")
    sell_model = load_model("rf_selling.joblib")
    sell_cols = load_model("selling_features.joblib")
except (RuntimeError, FileNotFoundError, EOFError, ValueError):
    rec_model = None
    yield_model = None
    sell_model = None
    yield_cols = []
    sell_cols = []

# PYDANTIC SCHEMAS
class LoginSchema(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)

class RecommendationSchema(BaseModel):
    username: str = "User"
    features: Dict[str, float]

class YieldSchema(BaseModel):
    username: str = "User"
    features: Dict[str, Any]

class SellingSchema(BaseModel):
    username: str = "User"
    features: Dict[str, Any]

class AdminQuerySchema(BaseModel):
    query: str = Field(min_length=1)

class HistoryCreateSchema(BaseModel):
    username: str = Field(min_length=1)
    action: str = Field(min_length=1)
    inputs: Dict[str, Any]
    result: str

class HistoryUpdateSchema(BaseModel):
    action: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    result: Optional[str] = None

# SECURITY HELPERS
def require_admin(
    x_admin_key: Optional[str] = Header(default=None),
) -> None:
    """
    Protect admin endpoints.
    If AGRISMART_ADMIN_API_KEY is not configured, the endpoint
    remains available for local development.
    """
    if ADMIN_API_KEY and x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Admin API key required",
        )
def require_model(model, name: str) -> None:
    """Ensure the requested ML model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{name} is not loaded. "
                "Run train_model.py first."
            ),
        )
# BASIC ENDPOINTS
@app.get("/")
def index():
    return {
        "status": "Online",
        "service": "Crop Prediction & CRUD FastAPI Engine",
    }
@app.get("/health")
def health():
    info = get_ml_information()

    return {
        "status": "healthy",
        "models_ready": info["models_ready"],
    }

# LOGIN
@app.post("/api/login")
def login(data: LoginSchema):
    role = verify_user(
        data.username,
        data.password,
    )
    if not role:
        raise HTTPException(
            status_code=401,
            detail="Invalid Username or Password",
        )
    return {
        "status": "success",
        "username": data.username,
        "role": role,
    }
# ADMIN - ML INFORMATION
@app.get(
    "/api/admin/ml-models",
    dependencies=[Depends(require_admin)],
)
def get_ml_models():
    result = get_ml_information()

    if result["data"] is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "ML metadata not found. "
                "Run train_model.py first."
            ),
        )

    return result
@app.get(
    "/api/admin/ml-status",
    dependencies=[Depends(require_admin)],
)
def get_ml_status():
    result = get_ml_information()

    return {
        "status": "success",
        "models_ready": result["models_ready"],
        "model_status": result["model_status"],
    }

# HISTORY - CREATE
@app.post(
    "/api/history",
    status_code=201,
)
def create_history(data: HistoryCreateSchema):
    record_id = log_history(
        data.username,
        data.action,
        data.inputs,
        data.result,
    )
    return {
        "status": "success",
        "message": "Record created",
        "id": record_id,
    }

# HISTORY - READ ALL
@app.get("/api/history")
def read_all_history(
    username: Optional[str] = Query(default=None),
):
    return {
        "status": "success",
        "data": get_all_history(username),
    }

# HISTORY - READ ONE
@app.get("/api/history/{record_id}")
def read_single_history(
    record_id: int = ApiPath(..., ge=1),
):
    record = get_history_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail="Record not found",
        )
    return {
        "status": "success",
        "data": record,
    }

# HISTORY - UPDATE
@app.put("/api/history/{record_id}")
def update_history(
    data: HistoryUpdateSchema,
    record_id: int = ApiPath(..., ge=1),
):
    updated = update_history_record(
        record_id,
        data.action,
        data.inputs,
        data.result,
    )
    if not updated:
        raise HTTPException(
            status_code=400,
            detail="Update failed or record not found",
        )
    return {
        "status": "success",
        "message": (
            f"Record {record_id} updated successfully"
        ),
    }

# HISTORY - DELETE
@app.delete("/api/history/{record_id}")
def delete_history(
    record_id: int = ApiPath(..., ge=1),
):
    removed = remove_history(record_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail="Record not found or already deleted",
        )
    return {
        "status": "success",
        "message": (
            f"Record {record_id} deleted successfully"
        ),
    }

# CROP RECOMMENDATION
@app.post("/api/predict/crop")
def predict_crop(data: RecommendationSchema):
    require_model(
        rec_model,
        "Recommendation model",
    )
    try:
        expected = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        missing = [
            column
            for column in expected
            if column not in data.features
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        input_df = pd.DataFrame(
            [[data.features[column] for column in expected]],
            columns=expected,
        )
        prediction = str(
            rec_model.predict(input_df)[0]
        )
        log_history(
            data.username,
            "Crop Recommendation",
            data.features,
            prediction,
        )
        return {
            "status": "success",
            "recommended_crop": prediction,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# CROP YIELD PREDICTION
@app.post("/api/predict/yield")
def predict_yield(data: YieldSchema):
    require_model(
        yield_model,
        "Yield model",
    )
    try:
        expected = [
            "Crop",
            "Season",
            "State",
            "Area",
            "Annual_Rainfall",
            "Fertilizer",
            "Pesticide",
        ]
        missing = [
            column
            for column in expected
            if column not in data.features
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        input_df = pd.DataFrame(
            [data.features]
        )
        categorical_columns = [
            column
            for column in [
                "Crop",
                "Season",
                "State",
            ]
            if column in input_df.columns
        ]
        encoded = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=True,
        )
        encoded = encoded.reindex(
            columns=yield_cols,
            fill_value=0,
        )
        value = round(
            float(
                yield_model.predict(encoded)[0]
            ),
            2,
        )
        log_history(
            data.username,
            "Yield Prediction",
            data.features,
            f"{value} Ton/Ha",
        )
        return {
            "status": "success",
            "predicted_yield_ton_per_ha": value,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# SELLING PRICE PREDICTION
@app.post("/api/predict/selling")
def predict_selling(data: SellingSchema):
    require_model(
        sell_model,
        "Selling price model",
    )
    try:
        expected = [
            "Crop",
            "State",
            "Season",
            "Quantity (kg)",
        ]
        missing = [
            column
            for column in expected
            if column not in data.features
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        quantity = float(
            data.features["Quantity (kg)"]
        )
        if quantity < 0:
            raise HTTPException(
                status_code=422,
                detail="Quantity (kg) cannot be negative",
            )
        input_df = pd.DataFrame(
            [data.features]
        )
        categorical_columns = [
            column
            for column in [
                "Crop",
                "Season",
                "State",
            ]
            if column in input_df.columns
        ]
        encoded = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=True,
        )
        encoded = encoded.reindex(
            columns=sell_cols,
            fill_value=0,
        )
        price = max(
            0.0,
            round(
                float(
                    sell_model.predict(encoded)[0]
                ),
                2,
            ),
        )
        revenue = round(
            price * quantity,
            2,
        )
        log_history(
            data.username,
            "Selling Forecast",
            data.features,
            f"₹ {price}/kg | Revenue ₹ {revenue}",
        )
        return {
            "status": "success",
            "predicted_price_per_kg": price,
            "predicted_total_value": revenue,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# ADMIN SQL QUERY
@app.post(
    "/api/admin/query",
    dependencies=[Depends(require_admin)],
)
def admin_query(data: AdminQuerySchema):
    return {
        "status": "success",
        "result": execute_admin_sql(data.query),
    }
