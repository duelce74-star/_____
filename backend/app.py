from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your superkart model
model_path = os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
model = joblib.load(model_path)

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Superkart ML Backend is running!"}

@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
