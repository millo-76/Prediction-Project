# League of Legends Match Predictor

Full-stack machine learning application that predicts the winner of a professional League of Legends match from historical team performance data (Oracle's Elixir).

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)

---

## Overview

Given two teams, the system returns:

- Predicted winner (blue or red)
- Blue-side win probability
- Red-side win probability

The project currently includes:

- A FastAPI backend for inference
- A React + TypeScript frontend for team-vs-team predictions
- Reproducible ML scripts for train/evaluate/predict
- A phase-based feature pipeline up to Phase 4 (ELO-enhanced)

---

## Current Status (Phase 4 Baseline)

| Component | Status |
|---|---|
| Data pipeline (2025 + 2026) | ✅ Complete |
| Baseline model (Phase 1) | ✅ Complete |
| Rolling-form model (Phase 3) | ✅ Complete |
| ELO-enhanced model (Phase 4) | ✅ Complete |
| Prediction API | ✅ Complete |
| Team lookup endpoint | ✅ Complete |
| Frontend UI (React) | ✅ Complete |

Latest logged result (chronological split):

- Model: Logistic Regression
- Dataset: `data/processed/phase4_elo_features.csv`
- Features: `wr_diff`, `blue_team_games`, `red_team_games`, `wr_diff_5`, `elo_diff`
- Accuracy: **0.6737**

Source: `experiments/results.csv` (row `phase4_logreg_elo`, 2026-05-24).

---

## Model

### Target

`blue_side_win` (binary):

- `1` = blue side wins
- `0` = red side wins

### Features (Phase 4)

| Feature | Description |
|---|---|
| `wr_diff` | Blue all-time win rate minus red all-time win rate |
| `blue_team_games` | Total games played by blue team |
| `red_team_games` | Total games played by red team |
| `wr_diff_5` | Blue rolling-5 win rate minus red rolling-5 win rate |
| `elo_diff` | Blue latest ELO minus red latest ELO |

### Training Setup

- Algorithm: Logistic Regression (`max_iter=1000`)
- Split: 80/20 chronological (no shuffle)
- Metrics: accuracy + classification report
- Primary artifact: `backend/ml/artifacts/logreg_phase4_elo.joblib`

---

## API

When backend is running:

- Swagger docs: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### `GET /`

Health message:

```json
{ "message": "LoL Match Predictor API is running" }
```

### `GET /health`

Simple status check:

```json
{ "status": "ok" }
```

### `GET /teams`

Returns available team names inferred from the Phase 4 feature dataset.

### `POST /predict`

Predict from raw engineered features.

Request:

```json
{
  "wr_diff": 0.13,
  "blue_team_games": 38,
  "red_team_games": 41,
  "wr_diff_5": 0.2,
  "elo_diff": 42.0
}
```

Response:

```json
{
  "predicted_class": 1,
  "predicted_winner": "blue",
  "blue_win_probability": 0.713,
  "red_win_probability": 0.287
}
```

### `POST /predict-from-teams`

Predict from team names; backend computes features automatically.

Request:

```json
{
  "blue_team": "T1",
  "red_team": "Gen.G"
}
```

Response shape:

```json
{
  "predicted_class": 1,
  "predicted_winner": "blue",
  "blue_win_probability": 0.64,
  "red_win_probability": 0.36,
  "blue_team": "T1",
  "red_team": "Gen.G",
  "features_used": {
    "wr_diff": 0.08,
    "blue_team_games": 42,
    "red_team_games": 39,
    "wr_diff_5": 0.2,
    "elo_diff": 31.5
  }
}
```

### `GET /model-info`

Returns active model metadata used by the API.

---

## Local Setup

Run all backend commands from project root.

### 1) Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install backend dependencies

The `backend/requirements.txt` file is currently empty, so install core packages directly:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic
```

### 3) Start backend API

```bash
uvicorn backend.app.main:app --reload
```

### 4) Start frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

---

## ML Workflow

All commands from project root:

```bash
python backend/ml/train.py
python backend/ml/evaluate.py
python backend/ml/predict.py
```

What each script does:

- `backend/ml/train.py`: trains Phase 4 logistic regression and saves artifact
- `backend/ml/evaluate.py`: evaluates artifact on random + chronological splits and logs results
- `backend/ml/predict.py`: single local prediction smoke test

---

## Project Structure

```text
backend/
  app/
    main.py
    models/
    routes/
    services/
  ml/
    artifacts/
    evaluate.py
    predict.py
    preprocess.py
    train.py
    features/
data/
  raw/
  processed/
frontend/
  src/
notebooks/
docs/
experiments/
reports/
```

---

## Data Source

Data comes from [Oracle's Elixir](https://oracleselixir.com/), filtered for team-level complete matches.

Current working files are stored in:

- `data/raw/`
- `data/processed/`

---

## Roadmap

- [x] Baseline pipeline and model
- [x] Historical integration (2025 + 2026)
- [x] Rolling-form feature set
- [x] ELO feature integration
- [x] Full-stack prediction flow (FastAPI + React)
- [ ] Add richer contextual features (patch, region, league strength)
- [ ] Compare non-linear models (XGBoost/LightGBM)
- [ ] Add model monitoring and automated retraining workflow
