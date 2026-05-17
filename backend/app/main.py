from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.predict import router as predict_router


app = FastAPI(title="LoL Match Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "LoL Match Predictor API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(predict_router)