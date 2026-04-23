from fastapi import FastAPI

from backend.app.routes.predict import router as predict_router


app = FastAPI(title="LoL Match Predictor API")


@app.get("/")
def read_root():
    return {"message": "LoL Match Predictor API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(predict_router)