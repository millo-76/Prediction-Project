from pydantic import BaseModel


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