# League of Legends Match Predictor

A machine learning web application that predicts the likely winner of a professional League of Legends match using historical match data.

## Current Phase
Phase 1: planning, data collection, and preprocessing

## MVP
The first version predicts match outcome using:
- blue team
- red team
- patch
- region
- side-based and recent team performance features

## Planned Stack
- Python
- pandas
- scikit-learn
- FastAPI
- React
- TypeScript

## Project Goals
- build a repeatable data pipeline
- train a baseline model
- expose predictions through an API
- later connect the model to a web frontend

## Directory Overview
- `backend/` - API and machine learning scripts
- `frontend/` - React frontend
- `data/raw/` - original downloaded datasets
- `data/processed/` - cleaned model-ready datasets
- `notebooks/` - EDA and experiments
- `docs/` - planning and design notes