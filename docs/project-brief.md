# League of Legends Match Predictor

## Project Overview

This project is a web-based League of Legends match predictor focused on professional play. The goal is to build a machine learning model that predicts the likely winner of a match using historical team performance, patch, region, and side-based features.

## Problem Statement

League of Legends match outcomes are often discussed using intuition, recent hype, or general team reputation. This project aims to create a more structured and data-driven way to evaluate matchups by training a predictive model on historical professional match data.

## MVP Goal

Build a baseline predictor that estimates the probability of the blue side or red side winning a professional League of Legends match.

## Initial Inputs

- Blue team
- Red team
- Patch
- Region

## Initial Output

- Predicted winner
- Blue-side win probability
- Red-side win probability

## Initial Scope

The first version will use historical professional match data only. It will focus on a small, reliable feature set and avoid live API ingestion, advanced draft logic, or deep player-level analytics until the baseline pipeline is working.

## Initial Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- React
- TypeScript

## Initial Model Plan

The first model will be a binary classification model using `blue_side_win` as the target variable. Logistic Regression will be used as the baseline model.

## Phase 1 Success Criteria

Phase 1 will be considered successful when the project has:

- a defined MVP
- a clean repo structure
- a raw dataset
- a preprocessing script
- a processed training dataset
- a documented baseline feature set

## Target Variable

The model predicts `blue_side_win`, a binary variable where:

- 1 = Blue side wins
- 0 = Red side wins

## Planned Data Expansion

After validating the current 2026-only pipeline, expand the dataset to include at least one previous full year of professional matches.

### Purpose

- improve feature stability
- reduce early-season cold start issues
- increase training volume
- support stronger rolling and historical team performance features

### Important constraints

- preserve strict date ordering
- compute all historical features sequentially
- avoid leakage from future matches
- account for patch/meta drift when evaluating results