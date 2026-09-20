from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

app = FastAPI(
    title="Real-Time ML Inference API",
    description="REST API for Machine Learning predictions"
)

# Train the model when the API starts
X = [
    [0.1, 0.2],
    [0.2, 0.3],
    [0.3, 0.4],
    [0.7, 0.6],
    [0.8, 0.7],
    [0.9, 0.8]
]
y = [0, 0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)


class PredictionInput(BaseModel):
    feature1: float
    feature2: float


@app.get("/")
def home():
    return {"message": "Real-Time ML Inference API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    features = [[data.feature1, data.feature2]]

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
