from fastapi import APIRouter, HTTPException, Request, status

from app.schema import GamePredictionRequest, GamePredictionResponse
from app.service import predict_game_success

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post(
    "/predict",
    response_model=GamePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Steam game pre-release success tier",
)
async def predict(
    payload: GamePredictionRequest,
    request: Request,
) -> GamePredictionResponse:
    model = getattr(request.app.state, "model", None)
    artifacts = getattr(request.app.state, "artifacts", None)

    if model is None or artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessing artifacts are not loaded.",
        )

    return predict_game_success(
        request=payload,
        model=model,
        artifacts=artifacts,
    )