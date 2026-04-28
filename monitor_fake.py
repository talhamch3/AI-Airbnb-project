import pandas as pd
import mlflow
import mlflow.pyfunc
import psycopg2
import os
import numpy as np
from datetime import datetime


# =========================================================
# 🔌 MLflow setup
# =========================================================
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Airbnb Price Prediction")


# =========================================================
# 📦 Load model
# =========================================================
def load_model():
    return mlflow.pyfunc.load_model("models:/AirbnbPriceModel/latest")


# =========================================================
# 📊 Load batch directly from CSV
# =========================================================
def get_batch():

    # Load pre-generated dataset
    df = pd.read_csv("data/monitoring_batch.csv")

    # Shuffle to simulate new incoming data
    batch = df.sample(n=100, random_state=None).copy()

    return batch


# =========================================================
# 🧠 Compute RMSE
# =========================================================
def compute_rmse(batch, model):

    features = ["bedrooms", "bathrooms", "availability_rate", "review_count"]

    X = batch[features]
    y = batch["price"]

    preds = model.predict(X)

    rmse = np.sqrt(((y - preds) ** 2).mean())

    return rmse


# =========================================================
# 🌦️ Seasonal metrics
# =========================================================
def compute_seasonal_metrics(batch):

    seasonal = (
        batch.groupby("season")["availability_rate"]
        .mean()
        .reset_index()
        .rename(columns={"availability_rate": "avg_availability"})
    )

    # Ensure all seasons exist
    all_seasons = ["Winter", "Spring", "Summer", "Autumn"]

    seasonal = (
        pd.DataFrame({"season": all_seasons})
        .merge(seasonal, on="season", how="left")
    )

    seasonal["avg_availability"] = seasonal["avg_availability"].fillna(0)

    return seasonal


# =========================================================
# 🏠 Listing-level metrics
# =========================================================
def compute_listing_metrics(batch):

    df = batch[["listing_id", "bedrooms", "availability_rate"]].copy()
    df = df.drop_duplicates(subset=["listing_id"])

    return df


# =========================================================
# 💾 Store in Postgres
# =========================================================
def store_in_postgres(rmse, seasonal_df, listing_df):

    run_time = datetime.now()

    conn = psycopg2.connect(
        host="postgres",
        database="airbnb",
        user="admin",
        password="admin"
    )

    cur = conn.cursor()

    # Tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_monitoring (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            rmse FLOAT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS seasonal_availability (
            id SERIAL PRIMARY KEY,
            season TEXT,
            avg_availability FLOAT,
            timestamp TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS listing_metrics (
            listing_id BIGINT PRIMARY KEY,
            bedrooms FLOAT,
            availability_rate FLOAT
        )
    """)

    # Insert RMSE
    cur.execute("""
        INSERT INTO model_monitoring (timestamp, rmse)
        VALUES (%s, %s)
    """, (run_time, float(rmse)))

    # Insert seasonal data
    for _, row in seasonal_df.iterrows():
        cur.execute("""
            INSERT INTO seasonal_availability (season, avg_availability, timestamp)
            VALUES (%s, %s, %s)
        """, (
            row["season"],
            float(row["avg_availability"]),
            run_time
        ))

    # Upsert listing metrics
    for _, row in listing_df.iterrows():
        cur.execute("""
            INSERT INTO listing_metrics (listing_id, bedrooms, availability_rate)
            VALUES (%s, %s, %s)
            ON CONFLICT (listing_id) DO UPDATE
            SET bedrooms = EXCLUDED.bedrooms,
                availability_rate = EXCLUDED.availability_rate
        """, (
            int(row["listing_id"]),
            float(row["bedrooms"]),
            float(row["availability_rate"])
        ))

    conn.commit()
    cur.close()
    conn.close()


# =========================================================
# 🚀 Main pipeline
# =========================================================
def run_monitoring():

    print("Loading model...")
    model = load_model()

    print("Loading batch...")
    batch = get_batch()

    print("Computing RMSE...")
    rmse = compute_rmse(batch, model)
    print("RMSE:", rmse)

    print("Computing seasonal metrics...")
    seasonal_df = compute_seasonal_metrics(batch)

    print("Computing listing metrics...")
    listing_df = compute_listing_metrics(batch)

    print("Saving to DB...")
    store_in_postgres(rmse, seasonal_df, listing_df)

    print("✅ Monitoring complete.")


# =========================================================
if __name__ == "__main__":
    run_monitoring()