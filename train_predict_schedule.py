import pandas as pd
import mlflow
import mlflow.pyfunc
import os
from datetime import datetime

from data_preprocessing import load_and_preprocess_data


# -------------------------------
# MLflow Setup
# -------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "AirbnbPriceModel"


# -------------------------------
# 1. Load Model (latest)
# -------------------------------
def load_model():
    model = mlflow.pyfunc.load_model("models:/AirbnbPriceModel@best")
    return model


# -------------------------------
# 2. Generate Batch Data
# -------------------------------
def generate_batch_data(n=50):

    X, y = load_and_preprocess_data()

    batch = X.sample(n=n, random_state=42)

    return batch


# -------------------------------
# 3. Predict
# -------------------------------
def predict_batch(model, batch_df):

    preds = model.predict(batch_df)

    result = batch_df.copy()
    result["predicted_price"] = preds

    return result


# -------------------------------
# 4. Save Results
# -------------------------------
def save_results(df):

    os.makedirs("outputs", exist_ok=True)

    filename = f"outputs/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    df.to_csv(filename, index=False)

    print("Saved:", filename)


# -------------------------------
# MAIN
# -------------------------------
def run_batch_pipeline():

    print("Loading model...")
    model = load_model()

    print("Generating batch data...")
    batch = generate_batch_data()

    print("Running predictions...")
    results = predict_batch(model, batch)

    print("Saving results...")
    save_results(results)

    print("Batch prediction completed.")


# -------------------------------
if __name__ == "__main__":
    run_batch_pipeline()