from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    TeamPredictionRequest,
    TeamPredictionResponse,
)
from backend.app.services.inference import predict_from_teams
from backend.ml.predict import predict


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict(request.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-from-teams", response_model=TeamPredictionResponse)
def make_team_prediction(request: TeamPredictionRequest) -> TeamPredictionResponse:
    try:
        result = predict_from_teams(request.blue_team, request.red_team)
        return TeamPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "artifact": "logreg_phase2_rolling.joblib",
        "features": ["wr_diff", "blue_team_games", "red_team_games", "wr_diff_5"],
        "target": "blue_side_win",
    }

@router.get("/teams")
def get_teams():
    teams = inference_service.get_available_teams()
    return {"teams": teams}

inference_service = __import__("backend.app.services.inference", fromlist=["get_available_teams"])