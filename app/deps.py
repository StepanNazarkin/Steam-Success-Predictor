from typing import Annotated, Any

import xgboost as xgb
from fastapi import Depends, HTTPException, Request, status


def get_artifacts(request: Request) -> dict[str, Any]:
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessing artifacts are not loaded.",
        )
    return artifacts


ArtifactsDep = Annotated[dict[str, Any], Depends(get_artifacts)]


def get_model(request: Request) -> xgb.XGBClassifier:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return model


ModelDep = Annotated[xgb.XGBClassifier, Depends(get_model)]
