from dill import load
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

with open("model.pkl", "rb") as f:  # Open the file in binary mode
    reloaded_model = pickle.load(f)

app = FastAPI()

class Payload(BaseModel):
    remote_ratio: int
    work_year: int
    experience_level: object
    employment_type: object
    employee_residence: object
    company_location: object
    company_size: object
    salary_in_usd: int

@app.post("/")
def predict(payload: Payload):
    df = pd.DataFrame([payload.dict()])  # Corrected 'model_dump()' to 'dict()'
    print(df)
    y_pred = reloaded_model.predict(df)
    response = {
        'prediction': y_pred[0],
        'model_name': 'rfg_model_v1',
        'model_last_updated': '2024_05_07',
    }
    return response
