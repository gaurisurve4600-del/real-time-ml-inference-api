# Real-Time ML Inference REST API
## Live API
https://real-time-ml-inference-api.onrender.com

## API Documentation
https://real-time-ml-inference-api.onrender.com/docs

## Project Overview
This project demonstrates a Real-Time Machine Learning Inference REST API built using FastAPI. It accepts JSON input and returns prediction results with probabilities.

## Features
- FastAPI REST API
- /predict endpoint
- JSON input and prediction output
- Input validation using Pydantic
- Unit testing using Pytest
- Docker containerization

## System Architecture
User/Application → REST API → FastAPI Server → Prediction Model → Prediction Response

## API Endpoint
POST /predict

Example Input:
{"feature1": 0.7, "feature2": 0.6}

The API returns the predicted class and probability.

## Files
- main.py - FastAPI application
- test_main.py - Unit tests
- requirements.txt - Python dependencies
- Dockerfile - Container configuration

## Run Application
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000

## Run Tests
pytest

## Conclusion
This project demonstrates an end-to-end ML inference system using FastAPI, REST API, testing, and Docker containerization.
