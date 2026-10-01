from fastapi import FastAPI
import pandas as pd

from src.data import BeneficiaryData
from src.model import load_or_train_model

app = FastAPI(title="Health Cost Prediction API")

model = load_or_train_model()



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