from fastapi import APIRouter, status

from app import service
from app.deps import ArtifactsDep, ModelDep
from app.schema import GamePredictionRequest, GamePredictionResponse

router = APIRouter(prefix="/steam-predictor", tags=["Prediction"])

@router.get("/metadata")
def get_metadata(artifacts: ArtifactsDep) -> dict[str, list[str]]:
    return service.get_metadata(artifacts)

@router.post(
    "/predict",
    response_model=GamePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Steam game pre-release success tier",
)
def predict(
    payload: GamePredictionRequest,
    model: ModelDep,
    artifacts: ArtifactsDep,
) -> GamePredictionResponse:
    return service.predict_game_success(
        request=payload,
        model=model,
        artifacts=artifacts,
    )
