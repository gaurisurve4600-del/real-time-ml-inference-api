from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post(
        "/predict",
        json={"feature1": 0.7, "feature2": 0.6}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_invalid_input():
    response = client.post(
        "/predict",
        json={"feature1": "wrong", "feature2": 0.5}
    )
    assert response.status_code == 422
