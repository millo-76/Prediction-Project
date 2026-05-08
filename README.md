# League of Legends Match Predictor

A full-stack machine learning application that predicts the winner of a professional League of Legends match using historical performance data from Oracle's Elixir.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)

---

## Overview

This project builds a pre-match win probability model for competitive LoL using team-level historical statistics. Given two teams, the model returns the predicted winner and a win probability for each side.

The project follows a structured, phase-based development approach — prioritizing a clean data pipeline, a reproducible training workflow, and a production-ready API before expanding to advanced modeling.

---

## Current Status — Phase 2 Complete

| Component | Status |
|---|---|
| Data pipeline (2025 + 2026) | ✅ Complete |
| Baseline model (Phase 1) | ✅ Complete |
| Rolling-feature model (Phase 2) | ✅ Complete |
| Prediction API | ✅ Complete |
| Team-name lookup endpoint | ✅ Complete |
| Frontend UI | 🔜 Planned (Phase 3) |
| Advanced feature engineering | 🔜 Planned (Phase 3) |

**Current model accuracy: ~64.34%** (Logistic Regression, 2025 + 2026 data, rolling features, chronological split)

---

## Model

### Target Variable

`blue_side_win` — binary classification

- `1` → Blue side wins
- `0` → Red side wins

### Features (Phase 2)

| Feature | Description |
|---|---|
| `wr_diff` | Blue team all-time win rate minus red team all-time win rate |
| `blue_team_games` | Total games played by blue team (all-time) |
| `red_team_games` | Total games played by red team (all-time) |
| `wr_diff_5` | Blue team rolling-5 win rate minus red team rolling-5 win rate |

**Key insights:**
- `wr_diff` remains the dominant predictor of match outcome.
- Adding `wr_diff_5` (short-term form) improved accuracy from ~60.6% to ~64.34% (chronological split).
- Integrating 2025 historical data stabilised win rate estimates and reduced early-season noise.

### Training Details

- Algorithm: Logistic Regression (`max_iter=1000`)
- Dataset: `data/processed/phase2_best_features.csv` (2025 + 2026, 3,815 matches)
- Train/test split: **80/20 chronological** — sorted by date, earliest 80% train / latest 20% test (no shuffling)
- Metrics: accuracy, precision, recall, F1-score
- Model artifact: `backend/ml/artifacts/logreg_phase2_rolling.joblib`
- Artifact records `split_strategy: "chronological"` for traceability

### Accuracy History

| Phase | Data | Features | Accuracy |
|---|---|---|---|
| Phase 1 | 2026 only | `wr_diff`, games | ~60.6% (random split) |
| Phase 2 | 2025 + 2026 | + `wr_diff_5` (rolling) | ~63.19% random / **~64.34% chrono** |

---

## API

The prediction API is served via FastAPI.

### `POST /predict`

Supply raw feature values directly.

```json
{
  "wr_diff": 0.13,
  "blue_team_games": 38,
  "red_team_games": 41,
  "wr_diff_5": 0.2
}
```

**Response**

```json
{
  "predicted_class": 1,
  "predicted_winner": "blue",
  "blue_win_probability": 0.713,
  "red_win_probability": 0.287
}
```

### `POST /predict-from-teams`

Supply team names — the API looks up historical stats automatically.

```json
{
  "blue_team": "T1",
  "red_team": "Gen.G"
}
```

**Response**

```json
{
  "predicted_class": 1,
  "predicted_winner": "blue",
  "blue_win_probability": 0.641,
  "red_win_probability": 0.359,
  "blue_team": "T1",
  "red_team": "Gen.G",
  "features_used": {
    "wr_diff": 0.08,
    "blue_team_games": 42,
    "red_team_games": 39,
    "wr_diff_5": 0.2
  }
}
```

### `GET /model-info`

Returns current model metadata (artifact name, features, target).

Interactive docs available at `/docs` when the server is running.

---

## Stack

| Layer | Technology |
|---|---|
| Data processing | Python, pandas |
| Machine learning | scikit-learn |
| API | FastAPI |
| Frontend | React, TypeScript |
| Model serialization | joblib |

---

## Backend — ML Scripts

All scripts are run from the project root.

| Script | Purpose | Command |
|---|---|---|
| `backend/ml/train.py` | Train the model on `phase2_best_features.csv` using a chronological split and save the artifact | `python backend/ml/train.py` |
| `backend/ml/evaluate.py` | Load the saved artifact and print accuracy + classification report for both a random split and a chronological split | `python backend/ml/evaluate.py` |
| `backend/ml/predict.py` | Load the artifact and run a single prediction (test input hardcoded at bottom) | `python backend/ml/predict.py` |

> `backend/ml/preprocess.py` is currently a placeholder — preprocessing logic lives in `notebooks/` and the feature pipeline in `backend/app/services/inference.py`.

### Artifacts

| File | Description |
|---|---|
| `logreg_baseline.joblib` | Phase 1 model — 2026-only data, 3 features (~60.6% accuracy) |
| `logreg_phase2_rolling.joblib` | Phase 2 model — 2025+2026 data, 4 features including rolling form (~64.34% accuracy, chronological split) |

---

## Project Structure

```
backend/
  app/          # FastAPI application (routes, schemas, config, inference)
  ml/           # Training, evaluation, and prediction scripts
    artifacts/  # Saved model files (.joblib)
data/
  raw/          # Original downloaded datasets (Oracle's Elixir CSVs)
  processed/    # Cleaned and feature-engineered datasets
frontend/       # React + TypeScript frontend (Phase 3)
notebooks/      # EDA, feature experiments, data refresh workflow
docs/           # Project briefs and weekly changelogs
```

---

## Data Source

Match data is sourced from [Oracle's Elixir](https://oracleselixir.com/), a community dataset covering professional League of Legends matches worldwide.

- Filtered to `datacompleteness == "complete"` and `position == "team"`
- Combined 2025 + 2026 season data — **3,815 unique matches** as of 2026-05-06
- Coverage: 2026-01-08 through 2026-05-06 (2026 season) + full 2025 season

---

## Phase 2 – Historical Data Integration ✅

**Goal:** Integrate 2025 Oracle's Elixir data to improve feature stability and reduce early-season cold start effects.

### What was done

- Downloaded and merged 2025 + 2026 Oracle's Elixir CSVs
- Rebuilt match-level dataset with strict chronological ordering
- Added rolling short-term form feature (`wr_diff_5`)
- Retrained Logistic Regression — accuracy improved from ~60.6% to ~64.34% (chronological split)
- Added `/predict-from-teams` endpoint (team name → auto feature lookup)
- Established `data_refresh.ipynb` workflow for future dataset updates

### Pipeline Rules

| Rule | Detail |
|---|---|
| Schema | Preserve common columns across both years |
| Ordering | Sort all matches chronologically before feature computation |
| Features | Compute sequentially — no lookahead |
| Leakage | No future data may influence past match features |
| Traceability | Retain `year` column for per-season comparison and debugging |

---

## Roadmap

- [x] Baseline pipeline and model (Phase 1)
- [x] 2025 + 2026 data integration and rolling features (Phase 2)
- [ ] Build React frontend — team picker, prediction display (Phase 3)
- [ ] Add patch-level and league/region features (Phase 3)
- [ ] Evaluate gradient boosting models — XGBoost, LightGBM (Phase 3)
- [ ] Head-to-head historical feature (Phase 3)
