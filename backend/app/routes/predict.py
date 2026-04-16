from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import PredictionRequest, PredictionResponse
from backend.ml.predict import predict


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict(request.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "features": ["wr_diff", "blue_team_games", "red_team_games"],
        "target": "blue_side_win",
    }