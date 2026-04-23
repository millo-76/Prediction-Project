# League of Legends Match Predictor

A full-stack machine learning application that predicts the winner of a professional League of Legends match using historical performance data from Oracle's Elixir.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)

---

## Overview

This project builds a pre-match win probability model for competitive LoL using team-level historical statistics. Given two teams, a patch, and a region, the model returns the predicted winner and a confidence probability.

The project follows a structured, phase-based development approach — prioritizing a clean data pipeline, a reproducible training workflow, and a production-ready API before expanding to advanced modeling.

---

## Current Status — Phase 1 Complete

| Component | Status |
|---|---|
| Data pipeline | ✅ Complete |
| Baseline model | ✅ Complete |
| Prediction API | ✅ Complete |
| Frontend UI | 🔄 In progress |
| 2025 data integration | 🔄 Planned |
| Advanced feature engineering | 🔄 Planned |

**Baseline model accuracy: ~60.6%** (Logistic Regression, 2026 data only)

---

## Model

### Target Variable

`blue_side_win` — binary classification

- `1` → Blue side wins
- `0` → Red side wins

### Features (Phase 1)

| Feature | Description |
|---|---|
| `wr_diff` | Blue team win rate minus red team win rate |
| `blue_team_games` | Total games played by blue team (all-time) |
| `red_team_games` | Total games played by red team (all-time) |

**Key insight:** Relative team strength (`wr_diff`) is the strongest single predictor of match outcome.

### Training Details

- Algorithm: Logistic Regression
- Train/test split: 80/20, `random_state=42`
- Metrics: accuracy, precision, recall, F1-score
- Model artifact: `backend/ml/artifacts/logreg_baseline.joblib`

---

## API

The prediction endpoint is served via FastAPI.

**`POST /predict`**

```json
{
  "blue_team_wr": 0.65,
  "red_team_wr": 0.52,
  "blue_team_games": 38,
  "red_team_games": 41
}
```

**Response**

```json
{
  "prediction": 1,
  "probability": 0.713,
  "predicted_winner": "Blue Side"
}
```

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

## Project Structure

```
backend/
  app/          # FastAPI application (routes, schemas, config)
  ml/           # Training, inference, preprocessing scripts
  artifacts/    # Saved model files
data/
  raw/          # Original downloaded datasets
  processed/    # Cleaned, model-ready datasets
frontend/       # React frontend (in progress)
notebooks/      # EDA and experiments
docs/           # Planning notes and changelogs
```

---

## Data Source

Match data is sourced from [Oracle's Elixir](https://oracleselixir.com/), a community dataset covering professional League of Legends matches worldwide.

- Filtered to `datacompleteness == "complete"` and `position == "team"`
- Currently uses 2026 season data; 2025 integration is planned next

---

## Phase 2 – Historical Data Integration

**Goal:** Integrate 2025 Oracle's Elixir data into the existing pipeline to improve feature stability and reduce early-season cold start effects.

### Pipeline Rules

| Rule | Detail |
|---|---|
| Schema | Preserve common columns across both years |
| Ordering | Sort all matches chronologically before feature computation |
| Features | Compute sequentially — no lookahead |
| Leakage | No future data may influence past match features |
| Traceability | Retain `year` column for per-season comparison and debugging |

### Evaluation Goals

- Compare model performance: 2026-only vs. 2025 + 2026
- Re-test rolling features with added historical context
- Measure whether a larger training window improves baseline Logistic Regression accuracy

---

## Roadmap

- [ ] Integrate 2025 season data for a larger training set
- [ ] Add patch-based and region-based features
- [ ] Evaluate gradient boosting models (XGBoost, LightGBM)
- [ ] Build and connect React frontend
- [ ] Add live team lookup and auto-filled predictions
