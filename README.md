# League of Legends Match Predictor

End-to-end project for predicting professional League of Legends match winners.
It includes a FastAPI backend, a React frontend, and reproducible ML scripts.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)

## What This Project Does

Given a matchup, the system predicts:

- Winner (blue side or red side)
- Blue-side win probability
- Red-side win probability

Two prediction modes are supported:

- Feature-based: provide engineered numeric features directly
- Team-based: provide team names, and backend computes matchup features

## Current Baseline

- Model: Logistic Regression
- Target: `blue_side_win`
- Current feature set: `wr_diff`, `blue_team_games`, `red_team_games`, `wr_diff_5`, `elo_diff`
- Main artifact: `backend/ml/artifacts/logreg_phase4_elo.joblib`
- Latest logged chronological accuracy: **0.6737**

Source: `experiments/results.csv` (latest `phase4_logreg_elo` row).

## Quickstart

Run from repository root unless otherwise noted.

### 1) Python Environment

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

If `backend/requirements.txt` is empty in your clone, install the core stack manually:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic
```

### 3) Start API

```bash
uvicorn backend.app.main:app --reload
```

API docs:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

### 4) Start Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:5173 and calls backend on http://127.0.0.1:8000.

## API Reference

### `GET /`

Returns service banner:

```json
{ "message": "LoL Match Predictor API is running" }
```

### `GET /health`

Returns API health:

```json
{ "status": "ok" }
```

### `GET /teams`

Returns the available team list inferred from `data/processed/phase4_elo_features.csv`.

### `GET /model-info`

Returns active model metadata and feature list.

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

Response shape:

```json
{
  "predicted_class": 1,
  "predicted_winner": "blue",
  "blue_win_probability": 0.713,
  "red_win_probability": 0.287
}
```

### `POST /predict-from-teams`

Predict from team names (backend computes feature vector).

Request:

```json
{
  "blue_team": "T1",
  "red_team": "Gen.G"
}
```

Response includes prediction plus derived features:

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

## Full Data-to-Prediction Workflow

Run from repository root.

### 1) Add New Raw Match Data

Place/update Oracle's Elixir exports in `data/raw/`.

Example files:

- `data/raw/2025_LoL_OraclesElixir.csv`
- `data/raw/2026_LoL_OraclesElixir.csv`

### 2) Rebuild Combined Match-Level + Base Feature Data (Script)

```bash
python backend/ml/preprocess.py
```

This regenerates:

- `data/processed/combined_processed_matches.csv`
- `data/processed/combined_feature_matches.csv`

### 3) Rebuild Phase 3 Rolling Features (Script)

```bash
python backend/ml/build_rolling_features.py
```

This regenerates:

- `data/processed/phase3_rolling_features.csv`

### 4) Build Phase 4 ELO Features

```bash
python backend/ml/features/elo_features.py
```

This reads `data/processed/phase3_rolling_features.csv` and writes:

- `data/processed/phase4_elo_features.csv`

### 5) Train the Production Model

```bash
python backend/ml/train.py
```

Behavior:

- Trains logistic regression (`max_iter=1000`)
- Uses chronological split (`80/20`, no shuffle)
- Saves artifact to `backend/ml/artifacts/logreg_phase4_elo.joblib`

### 6) Evaluate and Log Experiment Metrics

```bash
python backend/ml/evaluate.py
```

Behavior:

- Evaluates chronological split (official metric) and random split (diagnostic only)
- Prints classification reports
- Appends experiment row to `experiments/results.csv`

Model promotion policy:

- Promoted model selection uses chronological evaluation only.
- Random split output is diagnostic and is not used for model promotion.

### 7) (Optional) Local Prediction Smoke Test

```bash
python backend/ml/predict.py
```

### 8) Use Downloaded Web Schedule Export (Default)

Download the latest schedule export from the web and place it at:

- Direct CSV download link: https://lol.fandom.com/wiki/Special:CargoExport?tables=MatchSchedule&fields=OverviewPage,Team1,Team2,DateTime_UTC,DateTime_UTC__precision,Patch&where=DateTime_UTC%20%3E%3D%20NOW()%20AND%20(OverviewPage%20LIKE%20%27%25LCK%25%27%20OR%20OverviewPage%20LIKE%20%27%25LPL%25%27%20OR%20OverviewPage%20LIKE%20%27%25LEC%25%27%20OR%20OverviewPage%20LIKE%20%27%25LTA%25%27%20OR%20OverviewPage%20LIKE%20%27%252026%20Mid-Season%20Invitational%25%27)&order_by=DateTime_UTC%20ASC&limit=200&format=csv

- `data/raw/schedule_export.csv`

Then normalize it into the pipeline input format:

```bash
python backend/scripts/scrape_schedule.py
```

Default output:

- `data/processed/schedule.csv`

### 9) Generate Batch Predictions

```bash
python backend/scripts/batch_predict.py
```

Default input/output:

- Input: `data/processed/schedule.csv`
- Output: `data/predictions/schedule_predictions.csv`

### 10) Update Prediction Results as Matches Finish

```bash
python backend/ml/update_prediction_results.py --from-oracles-elixir
```

### 11) Build Coverage Reports

```bash
python backend/scripts/coverage_summary.py
```

### 12) One-Command Pipeline Modes

Use the orchestrator script to run common pipeline flows:

```bash
python backend/scripts/run_pipeline.py --mode full-retrain
python backend/scripts/run_pipeline.py --mode predict-only
python backend/scripts/run_pipeline.py --mode sync-only
```

Notes:

- `full-retrain` runs preprocess -> rolling features -> elo features -> train -> evaluate -> batch predict -> sync -> coverage.
- `predict-only` runs batch prediction + coverage reporting.
- `sync-only` runs Oracle's Elixir result sync + coverage reporting.

## Data and Artifacts

- Raw data: `data/raw/`
- Processed datasets: `data/processed/`
- Model artifacts: `backend/ml/artifacts/`
- Experiment history: `experiments/results.csv`
- Prediction logs: `data/predictions/`

Primary data source: [Oracle's Elixir](https://oracleselixir.com/).

## Repository Layout

```text
backend/
  app/
    main.py                 # FastAPI app entrypoint
    routes/predict.py       # API routes
    services/inference.py   # Team matchup feature construction
    models/schemas.py       # Pydantic request/response models
  ml/
    train.py
    evaluate.py
    predict.py
    prediction_logger.py
    update_prediction_results.py
    artifacts/
  scripts/
    scrape_schedule.py
    batch_predict.py
frontend/
  src/
data/
  raw/
  processed/
  predictions/
  cache/
experiments/
reports/
notebooks/
docs/
```

## Troubleshooting

### `ModuleNotFoundError` when running scripts

- Run commands from repository root.
- Ensure virtual environment is activated.

### API starts but frontend cannot predict

- Confirm backend is running at `127.0.0.1:8000`.
- Confirm frontend runs at `localhost:5173`.
- Check browser network tab for failed requests to `/teams` or `/predict-from-teams`.

### Team not found for prediction

- Team names must match values returned by `GET /teams`.
- Verify `data/processed/phase4_elo_features.csv` exists and includes that team.

### Schedule source preference

- Default workflow uses downloaded web schedule exports saved to `data/raw/schedule_export.csv`.
- The pipeline then normalizes that file with `python backend/scripts/scrape_schedule.py` into `data/processed/schedule.csv`.

## Roadmap

- [x] Phase 1 baseline
- [x] Phase 3 rolling-form features
- [x] Phase 4 ELO-enhanced features
- [x] API + frontend integration
- [ ] Full workflow UI integration (trigger schedule import, batch predictions, and result updates from frontend)
- [ ] Richer context features (patch, league strength)
- [ ] Alternative models (XGBoost/LightGBM)
- [ ] Monitoring + retraining automation
