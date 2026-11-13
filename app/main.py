from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import xgboost as xgb
from fastapi import FastAPI

from app.router import router

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / "best_xgboost.json")

    app.state.model = model
    app.state.artifacts = joblib.load(MODELS_DIR / "preprocessing_artifacts.joblib")

    yield

    app.state.model = None
    app.state.artifacts = None


app = FastAPI(
    title="Steam Game Success Prediction API",
    description="Pre-release Steam game success tier classification service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str | bool]:
    is_ready = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": is_ready}
