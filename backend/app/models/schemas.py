from pydantic import BaseModel
from typing import Dict, Union


class PredictionRequest(BaseModel):
    wr_diff: float
    blue_team_games: int
    red_team_games: int
    wr_diff_5: float


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_winner: str
    blue_win_probability: float
    red_win_probability: float


class TeamPredictionRequest(BaseModel):
    blue_team: str
    red_team: str


class TeamPredictionResponse(PredictionResponse):
    blue_team: str
    red_team: str
    features_used: Dict[str, Union[float, int]]