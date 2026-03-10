import pandas as pd
import traceback
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Set tracking URI to your MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
app = FastAPI(title="Airbnb Price Prediction API")

model_client = None
MODEL_NAME = "AirbnbPriceModel"

class Listing(BaseModel):
    bedrooms: float
    bathrooms: float
    availability_rate: float
    review_count: float

def get_model():
    global model_client
    if model_client is None:
        print("Loading MLflow model...")
        try:
            model_client = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
            print("Model loaded successfully!")
        except Exception as e:
            print("Error loading model:", str(e))
            traceback.print_exc()
            raise e
    return model_client

@app.post("/predict")
def predict(listing: Listing):
    try:
        model = get_model()
        input_df = pd.DataFrame([listing.dict()])
        prediction = model.predict(input_df)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        # Show the actual error for debugging
        print("Prediction error:", str(e))
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/example")
def example_input():
    return {
        "bedrooms": 2,
        "bathrooms": 1.5,
        "availability_rate": 0.8,
        "review_count": 10
    }