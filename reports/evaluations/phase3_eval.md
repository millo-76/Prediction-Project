# Phase 3 Rolling Logistic Regression Baseline

## Model

Logistic Regression

## Dataset

data/processed/phase3_rolling_features.csv

## Features

- wr_diff
- blue_team_games
- red_team_games
- wr_diff_5

## Evaluation Method

Chronological train/test split

Train Period:
2025-01-11 -> 2026-03-03

Test Period:
2026-03-03 -> 2026-05-16

## Results

Accuracy: 65.29%

### Classification Report

=== Model Evaluation Setup ===

- Artifact: backend\ml\artifacts\logreg_phase3_rolling.joblib
- Dataset: data\processed\phase3_rolling_features.csv
- Features: ['wr_diff', 'blue_team_games', 'red_team_games', 'wr_diff_5']
- Target: blue_side_win
- Test size: 0.2
- Random state: 42

=== Random Split Evaluation ===

- Accuracy: 0.6226

Classification Report:

                           precision    recall  f1-score   support

                       0       0.63      0.52      0.57      1299
                       1       0.62      0.71      0.66      1406

                accuracy                           0.62      2705

               macro avg       0.62      0.62      0.62      2705

            weighted avg       0.62      0.62      0.62      2705

- Train period: 2025-01-11 -> 2026-03-03
- Test period:  2026-03-03 -> 2026-05-16
- Train rows: 10817
- Test rows: 2705

=== Chronological Split Evaluation ===

- Accuracy: 0.6529

Classification Report:

                        precision    recall  f1-score   support

                    0       0.65      0.55      0.60      1260
                    1       0.65      0.74      0.70      1445

             accuracy                           0.65      2705

            macro avg       0.65      0.65      0.65      2705
        
         weighted avg       0.65      0.65      0.65      2705

## Artifact

backend/ml/artifacts/logreg_phase3_rolling.joblib

## Notes

- Uses rolling momentum features
- Uses chronological evaluation to reduce leakage
- Dataset refreshed on 2026-05-16