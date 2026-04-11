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

- [ ] Define what one row represents (1 match)
- [ ] Finalize raw schema fields
- [ ] Map dataset columns → project schema
- [ ] Identify missing required fields
- [ ] Decide how to handle:
  - [ ] missing values
  - [ ] inconsistent team names
  - [ ] patch formatting
- [ ] Update `docs/data-notes.md` with mapping decisions

---

## Day 3 – Data Cleaning Exploration

- [ ] Load dataset into notebook/script
- [ ] Normalize team names (basic pass)
- [ ] Inspect unique team values
- [ ] Check for duplicate matches
- [ ] Check for null values in key columns
- [ ] Filter out invalid or incomplete rows
- [ ] Verify target column (`blue_side_win`) can be created
- [ ] Document cleaning steps in notes

---

## Day 4 – Preprocessing Script (v1)

- [ ] Begin `backend/ml/preprocess.py`
- [ ] Load raw dataset from `data/raw/`
- [ ] Select only required columns
- [ ] Rename columns to match schema
- [ ] Create `blue_side_win` column
- [ ] Drop unnecessary columns
- [ ] Save output to `data/processed/processed_matches.csv`
- [ ] Run script successfully end-to-end

---

## Day 5 – Feature Planning

- [ ] Define baseline features:
  - [ ] blue_team
  - [ ] red_team
  - [ ] patch
  - [ ] region
  - [ ] (future) team win rates
- [ ] Decide how to calculate:
  - [ ] recent team win rate (rolling)
- [ ] Document feature definitions in `docs/data-notes.md`
- [ ] Plan how to avoid data leakage

---

## Day 6 – Model Planning

- [ ] Confirm model type: Logistic Regression
- [ ] Define target variable: `blue_side_win`
- [ ] Decide train/test split (e.g., 80/20)
- [ ] Decide encoding strategy:
  - [ ] team names (one-hot or label encoding)
  - [ ] patch
  - [ ] region
- [ ] Define evaluation metrics:
  - [ ] accuracy
  - [ ] precision
  - [ ] recall
- [ ] Outline `train.py` structure

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
