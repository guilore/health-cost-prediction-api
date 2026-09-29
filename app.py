import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("model.joblib")


@app.post("/predict")
def predict(data: dict):

    beneficiario = pd.DataFrame([data])

    probabilidade = model.predict_proba(beneficiario)[0, 1]

    high_cost = int(probabilidade >= 0.5)

    return {
        "probabilidade": probabilidade,
        "high_cost": high_cost
    }