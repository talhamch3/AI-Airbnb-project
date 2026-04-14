import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime

from data_preprocessing import load_and_preprocess_data


# MLflow connection
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# -------------------------------
# 1. Load Model
# -------------------------------
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# -------------------------------
# 2. Season Logic
# -------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# -------------------------------
# 3. Get Batch + Add Season
# -------------------------------
def get_batch():
    X, y = load_and_preprocess_data()

    batch = X.sample(n=50, random_state=42).copy()
    actual = y.loc[batch.index]

    # Create synthetic month (since dataset may not have one)
    batch["month"] = np.random.randint(1, 13, size=len(batch))

    # Map to seasons
    batch["season"] = batch["month"].apply(get_season)

    return batch, actual


# -------------------------------
# 4. Compute RMSE
# -------------------------------
def compute_metrics(actual, predicted):
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    return rmse


# -------------------------------
# 5. Seasonal Aggregation
# -------------------------------
def compute_seasonal_availability(batch_df):
    seasonal = (
        batch_df
        .groupby("season")["availability_rate"]
        .mean()
        .reset_index()
    )
    return seasonal


# -------------------------------
# 6. Store in Postgres
# -------------------------------
def store_in_postgres(rmse, seasonal_df):

    conn = psycopg2.connect(
        host="postgres",     # docker service name
        database="airbnb",
        user="admin",
        password="admin"
    )

    cur = conn.cursor()

    # RMSE table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            rmse FLOAT
        )
    """)

    # Seasonal table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS seasonal_availability (
            id SERIAL PRIMARY KEY,
            season TEXT,
            avg_availability FLOAT,
            timestamp TIMESTAMP
        )
    """)

    # Insert RMSE
    cur.execute("""
        INSERT INTO model_monitoring (timestamp, rmse)
        VALUES (%s, %s)
    """, (datetime.now(), float(rmse)))

    # Insert seasonal data
    for _, row in seasonal_df.iterrows():
        cur.execute("""
            INSERT INTO seasonal_availability (season, avg_availability, timestamp)
            VALUES (%s, %s, %s)
        """, (
            row["season"],
            float(row["availability_rate"]),
            datetime.now()
        ))

    conn.commit()
    cur.close()
    conn.close()


# -------------------------------
# 7. Main Pipeline
# -------------------------------
def run_monitoring():
    print("Loading model...")
    model = load_model()

    print("Getting batch...")
    X_batch, y_actual = get_batch()

    # -----------------------------
    # IMPORTANT FIX
    # -----------------------------
    # Keep monitoring columns separate
    X_monitor = X_batch.copy()

    # Drop non-model features BEFORE prediction
    X_model = X_batch.drop(columns=["month", "season"], errors="ignore")

    print("Predicting...")
    preds = model.predict(X_model)

    print("Computing RMSE...")
    rmse = compute_metrics(y_actual, preds)
    print("RMSE:", rmse)

    print("Computing seasonal availability...")
    seasonal_df = compute_seasonal_availability(X_monitor)

    print(seasonal_df)

    print("Saving to DB...")
    store_in_postgres(rmse, seasonal_df)

    print("Monitoring complete.")


# -------------------------------
if __name__ == "__main__":
    run_monitoring()