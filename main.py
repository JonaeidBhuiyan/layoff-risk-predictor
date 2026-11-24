from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "xgb_model.json"

logger = logging.getLogger("layoff-risk-api")
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Layoff Risk Predictor",
    description=(
        "An exploratory estimate of layoff risk "
        "from company, market and workforce data."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://layoff-risk-predictor-1.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
# ============================================================
# 3. LOAD PREPROCESSOR AND XGBOOST MODEL
# ============================================================

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Preprocessor file not found: {PREPROCESSOR_PATH}. "
        "Put preprocessor.pkl beside main.py."
    ) from exc
except Exception as exc:
    raise RuntimeError(
        "Could not load preprocessor.pkl. "
        "Check your scikit-learn version and requirements.txt. "
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc

try:
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
except FileNotFoundError as exc:
    raise RuntimeError(
        f"XGBoost model file not found: {MODEL_PATH}. "
        "Put xgb_model.json beside main.py."
    ) from exc
except Exception as exc:
    raise RuntimeError(
        "Could not load xgb_model.json. "
        "Check your XGBoost installation and model file. "
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc


# ============================================================
# 4. MODEL INPUT SCHEMA
# ============================================================

ORIGINAL_INPUT_COLUMNS = [
    "year", "quarter", "company", "industry", "region", "department",
    "economic_condition", "work_model", "inflation_rate", "interest_rate",
    "gdp_growth", "ai_adoption_score", "automation_level", "employees_hired",
    "layoffs", "attrition_rate", "employee_satisfaction", "avg_salary",
    "revenue_growth", "stock_growth", "productivity_index", "innovation_score",
    "industry_avg_salary", "ai_automation_interaction", "salary_per_employee",
]

MODEL_INPUT_COLUMNS = list(
    getattr(preprocessor, "feature_names_in_", ORIGINAL_INPUT_COLUMNS)
)


# ============================================================
# 5. GET CATEGORY OPTIONS FROM THE TRAINED ENCODER
# ============================================================

CATEGORY_NAMES = [
    "quarter", "industry", "region", "department",
    "economic_condition", "work_model",
]

try:
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    encoder = categorical_pipeline.named_steps["encode"]
    OPTIONS = {
        name: [str(value) for value in categories]
        for name, categories in zip(CATEGORY_NAMES, encoder.categories_)
    }
except Exception as exc:
    raise RuntimeError(
        "Could not extract dropdown categories from the preprocessor. "
        "Check the categorical pipeline and encoder step names."
    ) from exc


# ============================================================
# 6. GET TRAINING MEDIANS FOR OPTIONAL INPUTS
# ============================================================

NUMERIC_COLUMNS = [
    "year", "inflation_rate", "interest_rate", "gdp_growth",
    "employees_hired", "attrition_rate", "employee_satisfaction",
    "avg_salary", "revenue_growth", "stock_growth",
    "productivity_index", "innovation_score",
]

try:
    numeric_pipeline = preprocessor.named_transformers_["num"]
    numeric_imputer = numeric_pipeline.named_steps["impute"]
    numeric_medians = dict(zip(NUMERIC_COLUMNS, numeric_imputer.statistics_))
    TYPICAL = {
        "employee_satisfaction": float(numeric_medians["employee_satisfaction"]),
        "avg_salary": float(numeric_medians["avg_salary"]),
        "revenue_growth": float(numeric_medians["revenue_growth"]),
    }
    if not all(np.isfinite(value) for value in TYPICAL.values()):
        raise ValueError("One or more training medians are not finite.")
except Exception as exc:
    raise RuntimeError(
        "Could not read training medians from the numeric imputer. "
        "Check the numeric preprocessing pipeline."
    ) from exc


# ============================================================
# 7. PYDANTIC INPUT VALIDATION
# ============================================================

class CompanyData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    year: int = Field(ge=2005, le=2026)
    quarter: str
    industry: str
    region: str
    department: str
    economic_condition: str
    work_model: str

    inflation_rate: float = Field(ge=0, le=20)
    interest_rate: float = Field(ge=0, le=15)
    gdp_growth: float = Field(ge=-10, le=15)
    ai_adoption_score: int = Field(ge=0, le=100)
    automation_level: int = Field(ge=0, le=100)
    employees_hired: int = Field(ge=100, le=12000)
    attrition_rate: float = Field(ge=0, le=50)

    employee_satisfaction: Optional[float] = Field(default=None, ge=0, le=100)
    avg_salary: Optional[float] = Field(default=None, ge=0, le=250000)
    revenue_growth: Optional[float] = Field(default=None, ge=-50, le=80)
    stock_growth: float = Field(ge=-70, le=100)
    productivity_index: float = Field(ge=0, le=100)
    innovation_score: float = Field(ge=0, le=100)


# ============================================================
# 8. WHAT-IF ANALYSIS CONFIGURATION
# ============================================================

WHAT_IF = [
    ("ai_adoption_score", 15, "AI adoption +15", 0, 100),
    ("automation_level", 15, "Automation +15", 0, 100),
    ("attrition_rate", -3, "Attrition -3 pts", 0, 50),
    ("employee_satisfaction", 10, "Satisfaction +10", 0, 100),
    ("productivity_index", 10, "Productivity +10", 0, 100),
    ("innovation_score", 10, "Innovation +10", 0, 100),
    ("revenue_growth", 5, "Revenue growth +5 pts", -50, 80),
    ("stock_growth", 10, "Stock growth +10 pts", -70, 100),
]


# ============================================================
# 9. VALIDATE CATEGORICAL INPUTS
# ============================================================

def check_categories(data: CompanyData) -> None:
    for name, allowed_values in OPTIONS.items():
        value = getattr(data, name)
        if value not in allowed_values:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid {name}: {value}. "
                    f"Allowed values: {', '.join(allowed_values)}"
                ),
            )


# ============================================================
# 10. CONSTRUCT A MODEL INPUT ROW
# ============================================================

def make_row(data: dict) -> dict:
    row = dict(data)
    row.update({
        "company": "Unknown",
        "layoffs": np.nan,
        "industry_avg_salary": np.nan,
        "ai_automation_interaction": np.nan,
        "salary_per_employee": np.nan,
    })

    for key in ("employee_satisfaction", "avg_salary", "revenue_growth"):
        if row.get(key) is None:
            row[key] = np.nan

    missing_columns = [
        column for column in MODEL_INPUT_COLUMNS if column not in row
    ]
    if missing_columns:
        raise RuntimeError(
            "Missing model input columns: " + ", ".join(missing_columns)
        )

    return {column: row[column] for column in MODEL_INPUT_COLUMNS}


# ============================================================
# 11. PREPROCESS DATA AND MAKE PREDICTIONS
# ============================================================

def predict_scores(rows: list[dict]) -> list[float]:
    frame = pd.DataFrame(
        [make_row(row) for row in rows],
        columns=MODEL_INPUT_COLUMNS,
    )
    X_processed = preprocessor.transform(frame)
    dmatrix = xgb.DMatrix(X_processed)
    predictions = model.predict(dmatrix)
    return [
        float(np.clip(value, 0, 100))
        for value in np.asarray(predictions).reshape(-1)
    ]


# ============================================================
# 12. CONVERT SCORE INTO A RISK LEVEL
# ============================================================

def risk_level(score: float) -> str:
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 65:
        return "High"
    return "Critical"


# ============================================================
# 13. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "preprocessor": "loaded",
        "model": "XGBoost Booster",
        "model_file": MODEL_PATH.name,
        "model_features": model.num_features(),
    }


# ============================================================
# 14. OPTIONS ENDPOINT
# ============================================================

@app.get("/api/options")
def get_options():
    return OPTIONS


# ============================================================
# 15. MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/api/predict")
def predict(data: CompanyData):
    check_categories(data)
    base = data.model_dump()
    rows = [base]
    labels = []

    for field, delta, label, lower, upper in WHAT_IF:
        changed = dict(base)
        current = changed[field]
        if current is None:
            current = TYPICAL[field]
        changed[field] = min(upper, max(lower, current + delta))
        rows.append(changed)
        labels.append(label)

    try:
        scores = predict_scores(rows)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check model compatibility "
                "and input columns. "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    baseline = scores[0]
    drivers = [
        {"label": label, "change": round(new_score - baseline, 2)}
        for label, new_score in zip(labels, scores[1:])
    ]
    drivers.sort(key=lambda item: abs(item["change"]), reverse=True)

    return {
        "score": round(baseline, 1),
        "level": risk_level(baseline),
        "drivers": drivers,
        "note": (
            "This is an exploratory model score on a 0-100 scale, "
            "not a probability or a guarantee of layoffs. "
            "What-if changes are hypothetical model comparisons, "
            "not causal effects."
        ),
    }


# ============================================================
# 16. SERVE FRONTEND STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ============================================================
# 17. HOME PAGE
# ============================================================

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Layoff Risk Predictor API is running.",
        "docs": "/docs",
        "health": "/api/health",
        "frontend": "Place your frontend index.html inside the static folder.",
    }
