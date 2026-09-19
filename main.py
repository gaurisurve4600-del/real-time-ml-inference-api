from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time ML Inference API",
    description="REST API for Machine Learning Predictions"
)

class PredictionInput(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def home():
    return {"message": "Real-Time ML Inference API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Simple prediction logic for API demonstration
    score = (data.feature1 + data.feature2) / 2

    if score >= 0.5:
        prediction = 1
        probability = min(score, 1.0)
    else:
        prediction = 0
        probability = 1 - score

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
         }
