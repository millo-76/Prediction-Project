from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    TeamPredictionRequest,
    TeamPredictionResponse,
)
from backend.app.services.inference import predict_from_teams
from backend.ml.prediction_logger import log_prediction
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
        log_prediction(
            blue_team=request.blue_team,
            red_team=request.red_team,
            predicted_winner=result["predicted_winner"],
            blue_win_probability=result["blue_win_probability"],
            red_win_probability=result["red_win_probability"],
            model_name="Logistic Regression",
            model_artifact="logreg_phase4_elo.joblib",
        )
        return TeamPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "artifact": "logreg_phase4_elo.joblib",
        "features": ["wr_diff", "blue_team_games", "red_team_games", "wr_diff_5", "elo_diff"],
        "target": "blue_side_win",
    }

@router.get("/teams")
def get_teams():
    teams = inference_service.get_available_teams()
    return {"teams": teams}

inference_service = __import__("backend.app.services.inference", fromlist=["get_available_teams"])