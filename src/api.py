from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Health Cost Prediction API")

model = joblib.load("models/health_cost_model.joblib")


class BeneficiaryData(BaseModel):
    age: float
    er_visits_12m: float
    consultations_12m: float
    hospitalizations_12m: float
    exams_12m: float
    historical_cost: float


@app.get("/")
def home():
    return {
        "message": "Health Cost Prediction API is running"
    }


@app.post("/predict")
def predict(beneficiary: BeneficiaryData):

    data = pd.DataFrame([beneficiary.model_dump()])

    probability = model.predict_proba(data)[0, 1]

    return {
        "high_cost_probability": float(probability),
        "prediction": int(probability >= 0.5),
    }