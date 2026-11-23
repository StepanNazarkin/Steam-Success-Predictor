from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import xgboost as xgb
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.router import router

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = xgb.XGBClassifier(n_jobs=1)
    model.load_model(MODELS_DIR / "best_xgboost.json")

    app.state.model = model
    app.state.artifacts = joblib.load(MODELS_DIR / "preprocessing_artifacts.joblib")

    yield

    app.state.model = None
    app.state.artifacts = None


app = FastAPI(
    title="Steam Game Success Prediction API",
    description="Pre-release Steam game success tier classification service",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    is_ready = (
        getattr(app.state, "model", None) is not None
        and getattr(app.state, "artifacts", None) is not None
    )
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "model_loaded": False},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "model_loaded": True},
    )
