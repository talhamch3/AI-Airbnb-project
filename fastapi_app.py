import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI
from pydantic import BaseModel


# 1️⃣ Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2️⃣ Create FastAPI app
app = FastAPI(title="Airbnb Price Prediction API")

MODEL_NAME = "AirbnbPriceModel"
model_client = None


# 3️⃣ Input schema
class Listing(BaseModel):
    bedrooms: float
    bathrooms: float
    availability_rate: float
    review_count: float


# 4️⃣ Load model (lazy loading)
def get_model():
    global model_client
    if model_client is None:
        model_client = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
    return model_client


# 5️⃣ Prediction endpoint
@app.post("/predict")
def predict(listing: Listing):

    model = get_model()

    input_data = pd.DataFrame([listing.dict()])

    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}


# 6️⃣ Example endpoint
@app.get("/example")
def example():
    return {
        "bedrooms": 2,
        "bathrooms": 1.5,
        "availability_rate": 0.8,
        "review_count": 10
    }