# Week 1 Checklist – Phase 1

## Goal

Set up the data pipeline foundation: dataset selection, inspection, and preprocessing plan.

---

## Day 1 – Dataset Selection & Inspection

- [X] Choose initial dataset source (Oracle’s Elixir)
- [X] Download dataset (CSV or equivalent) - 2026_LoL_OraclesElixir.csv
- [X] Place file in `data/raw/`
- [X] Open dataset in notebook or script
- [X] Print first 5 rows (`df.head()`)
- [X] List all columns (`df.columns`)
- [X] Check data types (`df.info()`)
- [X] Identify key columns:
  - [X] match identifier
  - [X] date
  - [X] patch
  - [X] region
  - [X] blue team
  - [X] red team
  - [X] winner
- [X] Document:
  - [X] dataset source
  - [X] file name
  - [X] column list
  - [X] initial observations
  - [X] obvious issues (missing data, naming inconsistencies, etc.)

---

## Day 2 – Schema Definition

- [X] Load raw dataset
- [X] Filter to `datacompleteness == "complete"`
- [X] Filter to `position == "team"`
- [X] Keep only core columns:
  - [X] gameid
  - [X] date
  - [X] patch
  - [X] league
  - [X] side
  - [X] teamname
  - [X] result
- [X] Group rows by `gameid`
- [X] Confirm each grouped match has exactly 2 rows
- [X] Extract blue team row
- [X] Extract red team row
- [X] Create match-level records with:
  - [X] gameid
  - [X] date
  - [X] patch
  - [X] league
  - [X] blue_team
  - [X] red_team
  - [X] blue_side_win
- [X] Convert records into a new dataframe
- [X] Preview the new dataframe
- [X] Check for nulls or bad rows
- [X] Save to `data/processed/processed_matches.csv`
- [X] Document results in notes

---

## Day 3 Checklist

- [X] Sort dataset by date
- [X] Initialize team tracking dictionary
- [X] Loop through matches sequentially
- [X] Compute pre-match win rates
- [X] Add features:
  - [X] blue_team_wr
  - [X] red_team_wr
  - [X] blue_team_games
  - [X] red_team_games
- [X] Update stats after each match
- [X] Create feature dataframe
- [X] Validate values (0–1 range)
- [X] Save to processed file

---

## Day 4 Checklist

- [X] Load feature dataset
- [X] Add wr_diff feature
- [X] Define X (features) and y (target)
- [X] Perform train/test split
- [X] Train Logistic Regression model
- [X] Generate predictions
- [X] Evaluate accuracy
- [X] Print classification report
- [X] Inspect model coefficients
- [X] Test probability predictions

---

## Day 5 Checklist

- [X] Load processed match-level dataset
- [X] Sort matches by date
- [X] Set rolling window size to 5
- [X] Track each team’s recent results
- [X] Compute pre-match rolling win rates
- [X] Compute recent games counts
- [X] Create:
  - [X] blue_team_wr_5
  - [X] red_team_wr_5
  - [X] blue_team_games_5
  - [X] red_team_games_5
  - [X] wr_diff_5
- [X] Save updated feature dataset
- [X] Retrain Logistic Regression
- [X] Compare new accuracy to Day 4 baseline
- [X] Record results in notes

---

## Day 6 – Baseline Model Formalization

- [X] Confirm final Phase 1 feature set:
  - [X] wr_diff
  - [X] blue_team_games
  - [X] red_team_games
- [X] Confirm target variable:
  - [X] blue_side_win
- [X] Confirm train/test split:
  - [X] 80/20
  - [X] random_state=42
- [X] Confirm evaluation metrics:
  - [X] accuracy
  - [X] precision
  - [X] recall
  - [X] f1-score
- [X] Move model training logic into `backend/ml/train.py`
- [X] Load processed feature dataset inside `train.py`
- [X] Train Logistic Regression model inside script
- [X] Print evaluation results from script
- [X] Print coefficient table from script
- [X] Save trained model artifact
- [X] Save run notes / results summary

### Prediction Pipeline

- [ ] Create `predict.py`
- [ ] Load saved model artifact
- [ ] Define prediction function
- [ ] Accept feature inputs
- [ ] Return:
  - [ ] predicted class
  - [ ] probability
- [ ] Test prediction locally

---

## Day 7 – Review & Prep for Week 2

- [ ] Clean up folder structure if needed
- [ ] Verify:
  - [ ] raw dataset exists
  - [ ] preprocessing script runs
  - [ ] processed dataset is created
- [ ] Update README with current progress
- [ ] List known data issues
- [ ] Write Week 2 goals:
  - [ ] train baseline model
  - [ ] evaluate model
  - [ ] build first API endpoint

---

## End of Week 1 Definition of Done

- [ ] Dataset selected and stored
- [ ] Data inspected and documented
- [ ] Schema defined
- [ ] Preprocessing script created
- [ ] Processed dataset generated
- [ ] Feature plan documented
- [ ] Model approach defined
