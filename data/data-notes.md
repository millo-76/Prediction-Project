# Data Notes

## Data Source

Oracle’s Elixir – Professional League of Legends match dataset (2026 season)

## Overview

The dataset contains detailed match data at the player and team level. Each match is originally represented by multiple rows, requiring transformation into a match-level dataset for modeling.

---

## Raw Data Structure

- Each match contains **12 rows**:
  - 10 player-level rows
  - 2 team-level rows

- Key columns available:
  - `gameid`
  - `date`
  - `patch`
  - `league`
  - `side`
  - `teamname`
  - `result`
  - `position`
  - `datacompleteness`

---

## Data Filtering (Phase 1)

To ensure data quality and simplify processing:

- Only **complete matches** are used:

  ```python
  df = df[df["datacompleteness"] == "complete"]

  ## Phase 2 Model (2025 + 2026)

Accuracy: 62.63%

Improvement:
+2% over 2026-only baseline

Key Insight:
Adding historical data significantly improved model performance by reducing early-season noise and stabilizing win rate estimates.

Feature Importance:

- wr_diff remains the dominant predictor
- experience features decreased in importance

## Current Best Model

Artifact:

- `backend/ml/artifacts/logreg_phase2_rolling.joblib`

Features:

- `wr_diff`
- `blue_team_games`
- `red_team_games`
- `wr_diff_5`

Performance:

- Accuracy: 63.73%

Key insight:

- overall team strength is the strongest signal
- recent form adds meaningful predictive value when combined with long-term strength